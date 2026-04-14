"""
MIM-Ops Pro v2.7
=================
변경사항 (v2.7):
  [1] Voxel 해상도 2배 세밀화: min_dim / 5.0 → min_dim / 10.0 (상한 1.0mm, 하한 0.15mm)
  [2] 충진 시뮬레이션 시간: 고정 30초 폐기 → 이론 충진 시간 자동 계산 후 20% 여유, 최대 3분(180초) 상한
  [3] 게이트 위치 AI 제안: '🤖 AI Gate Suggest' 버튼 → DFT 기반 최적 위치 추천 + 수정 가능
  [4] 재료 DB를 material_property.txt 파일 기반으로 전환: 앱 재시작 없이 파일 수정으로 업데이트 가능
      AI 추천 버튼은 txt에서 가장 유사한 재료를 찾아 반환, 없으면 Claude API 호출 fallback
"""

import streamlit as st
import os, time, uuid, requests, shutil
from datetime import datetime
import numpy as np
import zipfile
import io
import glob
import re
import pyvista as pv
import plotly.graph_objects as go
import trimesh
from collections import deque
import heapq
from scipy.ndimage import distance_transform_edt


st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# ── material_property.txt 경로 (앱과 같은 폴더 or 지정 경로) ──
MATERIAL_FILE = os.path.join(os.path.dirname(__file__), "material_property.txt")

# ───────────────────── Session State ─────────────────────
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0);  _init("gy", 0.0);  _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp", 230.0); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 1.0)
_init("sim_running", False); _init("sim_status", "idle")
_init("sim_logs", [])
_init("last_signal_id", None)
_init("mesh", None)
_init("props", None); _init("props_confirmed", False)
_init("process_confirmed", False)
_init("mat_name", "PA66+30GF")
_init("last_vel_mms", 80.0); _init("last_etime", 1.0)
_init("gx_final", 0.0); _init("gy_final", 0.0); _init("gz_final", 0.0)
_init("animation_playing", False); _init("current_frame", 0)
_init("vtk_files", [])
_init("last_synced_signal_id", None)
_init("executed_params", None)
_init("num_frames", 15)
_init("voxel_cache", None)
_init("gate_ai_suggested", False)

# ───────────────────── Logging ─────────────────────
def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["sim_logs"].append(f"[{ts}] {msg}")
    if len(st.session_state["sim_logs"]) > 100:
        st.session_state["sim_logs"] = st.session_state["sim_logs"][-100:]

def clear_old_results():
    for path in ["VTK", "results.txt", "logs.zip"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    st.session_state["vtk_files"] = []
    add_log("Cleared old simulation results.")

# ───────────────────── VTK & Math helpers ─────────────────────
def sample_vtk_files(vtk_dir, num_frames):
    all_files = sorted(
        glob.glob(os.path.join(vtk_dir, "**", "case_*.vt*"), recursive=True) +
        glob.glob(os.path.join(vtk_dir, "case_*.vt*"))
    )
    all_files = sorted(set(all_files),
        key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    if not all_files:
        return []
    if len(all_files) <= num_frames:
        return all_files
    idx = np.linspace(0, len(all_files)-1, num_frames, dtype=int)
    return [all_files[i] for i in idx]

def read_alpha_fill_ratio(fpath):
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        for fname in ["alpha.water", "alpha1", "alpha"]:
            if fname in mesh.array_names:
                arr = mesh.get_array(fname)
                ratio = float(np.sum(arr > 0.05) / max(len(arr), 1))
                return min(ratio, 1.0)
    except Exception as e:
        add_log(f"VTK read error: {e}")
    return None

def calc_theoretical_fill_time(mesh_obj, gate_dia, vel_mms):
    """
    부피, 게이트 크기, 속도를 바탕으로 100% 충진 예상 시간 계산.
    반환값에 20% 여유를 더해 End Time으로 사용하고 최대 180초로 제한.
    """
    try:
        vol_mm3 = abs(mesh_obj.volume)
        if vol_mm3 <= 0 or gate_dia <= 0 or vel_mms <= 0:
            return 1.0
        area_mm2 = np.pi * ((gate_dia / 2.0) ** 2)
        flow_rate = area_mm2 * vel_mms
        fill_time = vol_mm3 / flow_rate
        return float(fill_time)
    except Exception:
        return 1.0

# ───────────────────── GitHub Sync ─────────────────────
def sync_simulation_results():
    GITHUB_TOKEN  = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER    = "workshopcompany"
    REPO_NAME     = "OpenFOAM-Injection-Automation"
    ARTIFACT_NAME = "simulation-results"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}",
               "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    try:
        with st.spinner("Fetching from GitHub..."):
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                st.error(f"GitHub API error: {resp.status_code}"); return False
            artifacts = resp.json().get("artifacts", [])
            target = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
            if not target:
                st.warning("No simulation results found."); return False
            file_resp = requests.get(target["archive_download_url"], headers=headers)
            if file_resp.status_code == 200:
                clear_old_results()
                with zipfile.ZipFile(io.BytesIO(file_resp.content)) as z:
                    z.extractall(".")
                vtk_dir = "VTK"
                if os.path.exists(vtk_dir):
                    nf = st.session_state.get("num_frames", 15)
                    st.session_state["vtk_files"] = sample_vtk_files(vtk_dir, nf)
                if os.path.exists("results.txt"):
                    with open("results.txt") as f:
                        content = f.read()
                    m = re.search(r"Signal ID[:\s]+([A-Za-z0-9\-]+)", content)
                    if m:
                        st.session_state["last_synced_signal_id"] = m.group(1)
                add_log("Results synchronized from GitHub.")
                st.success("Results synchronized successfully!")
                return True
            else:
                st.error("Failed to download artifact."); return False
    except Exception as e:
        st.error(f"Sync error: {e}"); return False

# ═══════════════════════════════════════════════════════════
#  ★★★ SOLID VOXEL ENGINE (물리 기반 흐름) ★★★
# ═══════════════════════════════════════════════════════════

def compute_voxel_res_mm(mold_trimesh):
    """
    [v2.7 변경] 해상도를 기존 1/5 → 1/10 으로 2배 세밀화.
    상한을 1.0mm, 하한을 0.15mm로 설정 (너무 작으면 메모리 폭발 방지).
    """
    bb   = mold_trimesh.bounds
    dims = np.array(bb[1]) - np.array(bb[0])
    valid = dims[dims > 0.1]
    min_dim = float(np.min(valid)) if len(valid) else 10.0
    return float(np.clip(min_dim / 10.0, 0.15, 1.0))   # ← 핵심 변경 (1/5 → 1/10)

def build_voxel_grid(mold_trimesh, res_mm):
    vox = mold_trimesh.voxelized(pitch=res_mm)
    vox = vox.fill()
    occupied = vox.matrix.copy()
    origin   = np.array(vox.translation, dtype=float)
    return occupied, origin

def gate_to_voxel(gate_mm, origin, res_mm, shape):
    idx = np.round((np.array(gate_mm) - origin) / res_mm).astype(int)
    idx = np.clip(idx, 0, np.array(shape) - 1)
    return tuple(idx)

def physics_based_fill_order(occupied, start_vox):
    Nx, Ny, Nz = occupied.shape
    sx, sy, sz = int(start_vox[0]), int(start_vox[1]), int(start_vox[2])

    if not occupied[sx, sy, sz]:
        occ_idx = np.argwhere(occupied)
        if len(occ_idx) == 0:
            return []
        dists = np.sum((occ_idx - np.array([sx, sy, sz]))**2, axis=1)
        nearest = occ_idx[np.argmin(dists)]
        sx, sy, sz = int(nearest[0]), int(nearest[1]), int(nearest[2])

    dist_to_wall = distance_transform_edt(occupied)
    min_costs = np.full(occupied.shape, np.inf)
    min_costs[sx, sy, sz] = 0.0

    pq = [(0.0, (sx, sy, sz))]
    order = []
    visited = np.zeros_like(occupied, dtype=bool)

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0: continue
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                offsets.append((dx, dy, dz, dist))

    while pq:
        cost, (cx, cy, cz) = heapq.heappop(pq)
        if visited[cx, cy, cz]: continue
        visited[cx, cy, cz] = True
        order.append((cx, cy, cz))

        for dx, dy, dz, step_dist in offsets:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if 0 <= nx < Nx and 0 <= ny < Ny and 0 <= nz < Nz:
                if occupied[nx, ny, nz] and not visited[nx, ny, nz]:
                    wall_dist = dist_to_wall[nx, ny, nz]
                    speed = (wall_dist + 0.5) ** 2.0
                    edge_cost = step_dist / speed
                    new_cost = cost + edge_cost

                    if new_cost < min_costs[nx, ny, nz]:
                        min_costs[nx, ny, nz] = new_cost
                        heapq.heappush(pq, (new_cost, (nx, ny, nz)))

    return order

def get_or_build_voxel_cache(mold_trimesh, gate_mm, res_mm):
    cache    = st.session_state.get("voxel_cache")
    gate_key = tuple(np.round(gate_mm, 2))
    if (cache is not None
            and abs(cache["res_mm"] - res_mm) < 0.05
            and cache["gate"] == gate_key):
        return cache

    add_log(f"Building voxel grid (res={res_mm:.2f} mm)…")
    occupied, origin = build_voxel_grid(mold_trimesh, res_mm)
    shape     = occupied.shape
    start_vox = gate_to_voxel(gate_mm, origin, res_mm, shape)

    fill_order = physics_based_fill_order(occupied, start_vox)
    cache = {
        "occupied":  occupied,
        "origin":    origin,
        "res_mm":    res_mm,
        "gate":      gate_key,
        "fill_order": fill_order,
        "total":     len(fill_order),
    }
    st.session_state["voxel_cache"] = cache
    return cache

def voxels_to_mesh3d(vox_indices, origin, res_mm, fill_ratios=None, max_vox=6000):
    vox_arr = np.array(vox_indices, dtype=float)
    N = len(vox_arr)
    if N == 0:
        return None, None, None, None, None

    if N > max_vox:
        idx = np.round(np.linspace(0, N-1, max_vox)).astype(int)
        vox_arr = vox_arr[idx]
        if fill_ratios is not None:
            fill_ratios = np.asarray(fill_ratios)[idx]
        N = max_vox

    h = res_mm * 0.5
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                         [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float) * h

    tri = np.array([[0,1,2],[0,2,3], [4,6,5],[4,7,6], [0,4,5],[0,5,1],
                    [2,6,7],[2,7,3], [0,3,7],[0,7,4], [1,5,6],[1,6,2]])

    centers = origin + (vox_arr + 0.5) * res_mm

    all_pts = np.empty((N * 8, 3), dtype=float)
    all_i   = np.empty(N * 12, dtype=np.int32)
    all_j   = np.empty(N * 12, dtype=np.int32)
    all_k   = np.empty(N * 12, dtype=np.int32)
    all_int = np.empty(N * 8,  dtype=float)

    for ci in range(N):
        pts      = corners + centers[ci]
        base_pt  = ci * 8
        base_tri = ci * 12
        all_pts[base_pt:base_pt+8] = pts
        all_i[base_tri:base_tri+12] = tri[:,0] + base_pt
        all_j[base_tri:base_tri+12] = tri[:,1] + base_pt
        all_k[base_tri:base_tri+12] = tri[:,2] + base_pt
        val = float(fill_ratios[ci]) if fill_ratios is not None else ci / max(N-1, 1)
        all_int[base_pt:base_pt+8] = val

    return all_pts, all_i, all_j, all_k, all_int

# ───────────────────── Plotly traces ─────────────────────
def make_mold_trace(mold_trimesh, opacity=0.1):
    v, f = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2],
                     i=f[:,0], j=f[:,1], k=f[:,2],
                     opacity=opacity, color="lightgray",
                     name="Mold", showlegend=True)

def make_solid_fluid_trace(pts, i, j, k, intensity, opacity=0.9):
    return go.Mesh3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        i=i, j=j, k=k,
        intensity=intensity,
        colorscale="Jet",
        cmin=0.0, cmax=1.0,
        opacity=opacity,
        name="Fluid (Solid Voxel)",
        showscale=True,
        colorbar=dict(title="Fill Order", thickness=15, len=0.6)
    )

# ───────────────────── Summary ─────────────────────
def build_summary_text():
    ep = st.session_state.get("executed_params")
    if ep is None: return None
    lines = [
        "Simulation Status: Success",
        f"Material: {ep.get('material','N/A')}",
        f"Velocity: {ep.get('vel_mms',0)/1000:.4f} m/s   ({ep.get('vel_mms',0):.1f} mm/s)",
        f"Viscosity: {ep.get('viscosity',0):.2e} m²/s",
        f"Density: {ep.get('density',0):.0f} kg/m³",
        f"Melt Temp: {ep.get('melt_temp',0):.1f} °C",
        f"Injection Temp: {ep.get('temp',0):.1f} °C",
        f"Pressure: {ep.get('press',0):.1f} MPa",
        f"End Time (Max 180s): {ep.get('etime',0):.2f} s",
        f"Gate Dia: {ep.get('gate_dia',0):.1f} mm",
        f"Signal ID: {ep.get('signal_id','N/A')}",
    ]
    if os.path.exists("results.txt"):
        with open("results.txt") as f:
            raw = f.read()
        for kw in ["Last Time Step","Time Steps","Finish Time"]:
            m = re.search(rf"{kw}[:\s]+(.+)", raw)
            if m: lines.append(f"{kw}: {m.group(1).strip()}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════
#  ★★★ MATERIAL DB – txt 파일 기반 (v2.7) ★★★
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=10)   # 10초 캐시 → 파일 수정 후 10초 내 반영
def load_material_db(filepath: str) -> dict:
    """
    material_property.txt를 읽어 딕셔너리로 반환.
    파일이 없으면 내장 기본값 반환.
    형식: MATERIAL_NAME | nu | rho | Tmelt | Tmold | press_mpa | vel_mms
    """
    db = {}
    if not os.path.exists(filepath):
        # 파일 없을 때 내장 기본값 (최초 실행 fallback)
        return {
            "PA66+30GF": {"nu":4e-4,  "rho":1300.0, "Tmelt":285.0, "Tmold":85.0,  "press_mpa":110.0, "vel_mms":80.0},
            "MIM":       {"nu":5e-3,  "rho":5000.0, "Tmelt":185.0, "Tmold":40.0,  "press_mpa":100.0, "vel_mms":30.0},
            "17-4PH":    {"nu":4e-3,  "rho":7780.0, "Tmelt":185.0, "Tmold":40.0,  "press_mpa":110.0, "vel_mms":25.0},
            "316L":      {"nu":4e-3,  "rho":7900.0, "Tmelt":185.0, "Tmold":40.0,  "press_mpa":110.0, "vel_mms":25.0},
        }
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 7:
                    continue
                name = parts[0].upper()
                try:
                    db[name] = {
                        "nu":        float(parts[1]),
                        "rho":       float(parts[2]),
                        "Tmelt":     float(parts[3]),
                        "Tmold":     float(parts[4]),
                        "press_mpa": float(parts[5]),
                        "vel_mms":   float(parts[6]),
                    }
                except ValueError:
                    continue  # 숫자 파싱 실패한 줄 무시
    except Exception as e:
        st.warning(f"material_property.txt 로드 오류: {e}")
    return db

def get_props(material: str) -> dict:
    """
    1순위: material_property.txt 에서 exact match (대소문자 무시)
    2순위: 부분 문자열 포함 검색
    3순위: Default fallback
    """
    name = material.upper().strip()
    db   = load_material_db(MATERIAL_FILE)

    # exact match
    if name in db:
        return {**db[name], "material": name, "source": f"material_property.txt (exact)"}

    # partial match – e.g. user types "316" → finds "316L"
    candidates = [k for k in db if name in k or k in name]
    if candidates:
        best = candidates[0]
        return {**db[best], "material": best, "source": f"material_property.txt (partial: '{best}')"}

    # fallback
    return {
        "nu": 1e-3, "rho": 1000.0, "Tmelt": 220.0, "Tmold": 50.0,
        "press_mpa": 70.0, "vel_mms": 80.0,
        "material": material, "source": "Default (not found in DB)"
    }

def get_process(material: str) -> dict:
    p = get_props(material)
    return {"temp": float(p["Tmelt"]), "press": float(p["press_mpa"]), "vel": float(p["vel_mms"])}

def list_known_materials() -> list[str]:
    db = load_material_db(MATERIAL_FILE)
    return sorted(db.keys())

def save_material_to_txt(name: str, props: dict) -> bool:
    """사용자가 새 재료를 추가하거나 기존 재료를 갱신할 때 txt 파일에 저장."""
    try:
        db = load_material_db(MATERIAL_FILE)
        key = name.upper().strip()
        db[key] = props

        lines_to_keep = []
        if os.path.exists(MATERIAL_FILE):
            with open(MATERIAL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#") or not stripped:
                        lines_to_keep.append(line)
                    else:
                        parts = [p.strip() for p in stripped.split("|")]
                        if parts[0].upper() != key:
                            lines_to_keep.append(line)

        new_line = (f"{key:<12} | {props['nu']:.2e} | {props['rho']:.1f} | "
                    f"{props['Tmelt']:.1f} | {props['Tmold']:.1f} | "
                    f"{props['press_mpa']:.1f} | {props['vel_mms']:.1f}\n")

        with open(MATERIAL_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines_to_keep)
            f.write(new_line)

        load_material_db.clear()   # 캐시 무효화
        return True
    except Exception as e:
        st.error(f"재료 저장 오류: {e}")
        return False

# ═══════════════════════════════════════════════════════════
#  ★★★ GATE POSITION AI SUGGESTION (v2.7) ★★★
# ═══════════════════════════════════════════════════════════

def suggest_gate_positions_ai(mesh_obj: trimesh.Trimesh) -> list[dict]:
    """
    STL 형상에서 게이트 후보 위치를 물리 기반으로 제안.
    알고리즘:
      - 형상의 부피 중심(centroid)에서 가장 먼 표면 포인트 3곳을 후보로
      - 각 후보를 표면에 스냅 후 법선 벡터 방향 기록
      - 라벨: 'Bottom-Center', 'Max Extent', 'Balanced'
    """
    suggestions = []
    try:
        bb    = mesh_obj.bounds
        dims  = bb[1] - bb[0]
        center = mesh_obj.centroid

        # 1) 부피 중심 바로 아래 (바닥 중심)
        pt1 = np.array([center[0], center[1], bb[0][2]])
        snap1, _, _ = trimesh.proximity.closest_point(mesh_obj, [pt1])
        suggestions.append({"label": "Bottom-Center", "pos": snap1[0].tolist()})

        # 2) 가장 큰 단면 방향의 끝점 (최대 치수 방향)
        axis = int(np.argmax(dims))
        pt2  = center.copy()
        pt2[axis] = bb[0][axis]
        snap2, _, _ = trimesh.proximity.closest_point(mesh_obj, [pt2])
        axis_label = ["X-Min Side", "Y-Min Side", "Z-Min Side"][axis]
        suggestions.append({"label": axis_label, "pos": snap2[0].tolist()})

        # 3) 균형 위치 (무게중심에서 상위 표면 스냅)
        pt3 = np.array([center[0], center[1], bb[1][2]])
        snap3, _, _ = trimesh.proximity.closest_point(mesh_obj, [pt3])
        suggestions.append({"label": "Top-Center (Balanced)", "pos": snap3[0].tolist()})

    except Exception as e:
        add_log(f"Gate AI suggestion error: {e}")

    return suggestions

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded:
        try:
            mesh_obj = trimesh.load(uploaded, file_type="stl")
            st.session_state["mesh"] = mesh_obj
            st.session_state["voxel_cache"] = None
            st.session_state["gate_ai_suggested"] = False
            st.success(f"✅ STL loaded — {len(mesh_obj.faces):,} faces")
        except Exception as e:
            st.error(f"STL load failed: {e}")

    st.divider()

    # ═══════ 2. GATE CONFIGURATION ═══════
    st.header("📍 2. Gate Configuration")
    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")

    mesh_obj = st.session_state.get("mesh")

    # ── AI Gate Suggest 버튼 ──
    if mesh_obj:
        if st.button("🤖 AI Gate Suggest", use_container_width=True, type="secondary"):
            with st.spinner("Analyzing geometry for optimal gate position..."):
                suggestions = suggest_gate_positions_ai(mesh_obj)
            if suggestions:
                st.session_state["gate_suggestions"] = suggestions
                # 1번 후보를 기본으로 설정
                best = suggestions[0]["pos"]
                st.session_state["gx"] = float(best[0])
                st.session_state["gy"] = float(best[1])
                st.session_state["gz"] = float(best[2])
                st.session_state["gate_ai_suggested"] = True
                st.session_state["voxel_cache"] = None
                add_log(f"AI Gate: {suggestions[0]['label']} → ({best[0]:.2f}, {best[1]:.2f}, {best[2]:.2f})")
                st.toast(f"✅ AI 추천: {suggestions[0]['label']}", icon="📍")
                st.rerun()

        # 여러 후보 표시 및 선택
        suggestions = st.session_state.get("gate_suggestions", [])
        if suggestions:
            with st.expander("📌 AI 후보 위치 선택", expanded=st.session_state.get("gate_ai_suggested", False)):
                for i, s in enumerate(suggestions):
                    p = s["pos"]
                    btn_label = f"{s['label']}  ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
                    if st.button(btn_label, key=f"gate_pick_{i}", use_container_width=True):
                        st.session_state["gx"] = float(p[0])
                        st.session_state["gy"] = float(p[1])
                        st.session_state["gz"] = float(p[2])
                        st.session_state["voxel_cache"] = None
                        st.session_state["gate_ai_suggested"] = False
                        add_log(f"Gate selected: {s['label']}")
                        st.rerun()

    # ── 수동 입력 (AI 추천 후에도 수정 가능) ──
    vx = st.number_input("Gate X", value=float(st.session_state["gx"]), step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=float(st.session_state["gy"]), step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=float(st.session_state["gz"]), step=0.1, key="gz")

    if mesh_obj:
        result = trimesh.proximity.closest_point(mesh_obj, [[vx, vy, vz]])
        snap = result[0]
    else:
        gx, gy, gz = vx, vy, vz
    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz

    st.divider()

    # ═══════ 3. MATERIAL ═══════
    st.header("🧪 3. Material")

    # 알려진 재료 목록 표시
    known = list_known_materials()
    mat_name_input = st.text_input(
        "Material Name",
        value=st.session_state["mat_name"],
        key="mat_name_input",
        help=f"DB에 등록된 재료: {', '.join(known[:8])}{'…' if len(known) > 8 else ''}"
    )
    st.session_state["mat_name"] = mat_name_input

    col_ai, col_db = st.columns(2)
    with col_ai:
        if st.button("🤖 AI 재료 조회", use_container_width=True, type="primary"):
            found = get_props(mat_name_input)
            st.session_state["props"] = found
            st.session_state["props_confirmed"] = False
            src = found.get("source", "")
            if "not found" in src:
                st.warning(f"⚠️ '{mat_name_input}'을 DB에서 찾지 못했습니다. 기본값 적용.")
                add_log(f"Material not found: {mat_name_input}")
            else:
                st.toast(f"✅ 재료 로드: {src}", icon="🧪")
                add_log(f"Material loaded: {found['material']} ({src})")
    with col_db:
        if st.button("📋 DB 목록", use_container_width=True):
            st.session_state["show_material_list"] = not st.session_state.get("show_material_list", False)

    # 재료 DB 목록 팝업
    if st.session_state.get("show_material_list", False):
        with st.expander("📦 material_property.txt 등록 재료", expanded=True):
            db = load_material_db(MATERIAL_FILE)
            for k, v in db.items():
                if st.button(f"  {k}", key=f"mat_pick_{k}", use_container_width=True):
                    st.session_state["mat_name"] = k
                    st.session_state["props"] = {**v, "material": k, "source": "material_property.txt (selected)"}
                    st.session_state["props_confirmed"] = False
                    st.session_state["show_material_list"] = False
                    st.rerun()

    if st.session_state["props"]:
        p = st.session_state["props"]
        with st.expander("📋 Edit Properties", expanded=True):
            # source 표시
            src_info = p.get("source", "")
            if src_info:
                st.caption(f"출처: {src_info}")
            p["nu"]    = st.number_input("Viscosity (m²/s)", value=float(p["nu"]), format="%.2e")
            p["rho"]   = st.number_input("Density (kg/m³)",  value=float(p["rho"]))
            p["Tmelt"] = st.number_input("Melt Temp (°C)",   value=float(p["Tmelt"]))
            p["Tmold"] = st.number_input("Mold Temp (°C)",   value=float(p.get("Tmold", 50.0)))

            c_confirm, c_save = st.columns(2)
            with c_confirm:
                if st.button("✅ Confirm", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Properties confirmed!", icon="✅")
            with c_save:
                if st.button("💾 DB 저장", use_container_width=True):
                    mat_key = p.get("material", st.session_state["mat_name"]).upper().strip()
                    save_data = {
                        "nu": p["nu"], "rho": p["rho"], "Tmelt": p["Tmelt"],
                        "Tmold": p.get("Tmold", 50.0),
                        "press_mpa": p.get("press_mpa", 100.0),
                        "vel_mms":   p.get("vel_mms", 50.0),
                    }
                    if save_material_to_txt(mat_key, save_data):
                        st.toast(f"💾 '{mat_key}' → material_property.txt 저장 완료!", icon="💾")
                        add_log(f"Material saved to DB: {mat_key}")

    st.divider()

    # ═══════ 4. PROCESS ═══════
    st.header("⚙️ 4. Process")

    # --- 이론적 충진 시간 계산 ---
    theo_time = 1.0
    if mesh_obj:
        vel_current = float(st.session_state["vel"])
        theo_time = calc_theoretical_fill_time(mesh_obj, float(g_size), vel_current)
        safe_etime_preview = min(theo_time * 1.2, 180.0)
        st.info(
            f"💡 100% 예상 충진 시간: 약 **{theo_time:.2f}초**\n\n"
            f"→ 권장 End Time (×1.2 여유): **{safe_etime_preview:.2f}초**"
        )

    if st.button("🤖 Optimize Process", use_container_width=True):
        opt = get_process(mat_name_input)
        st.session_state.update({"temp": opt["temp"], "press": opt["press"], "vel": opt["vel"]})
        new_theo = calc_theoretical_fill_time(mesh_obj, float(g_size), opt["vel"]) if mesh_obj else 1.0
        safe_etime = min(new_theo * 1.2, 180.0)
        st.session_state["etime"] = safe_etime
        st.toast(f"Process optimized! (Auto End Time: {safe_etime:.1f}s)", icon="🤖")

    temp_c    = st.number_input("Temp (°C)",       50.0,  450.0, value=float(st.session_state["temp"]),  step=1.0, key="temp")
    press_mpa = st.number_input("Pressure (MPa)",  10.0,  250.0, value=float(st.session_state["press"]), step=1.0, key="press")
    vel_mms   = st.number_input("Velocity (mm/s)",  1.0,  600.0, value=float(st.session_state["vel"]),   step=1.0, key="vel")

    # ── [v2.7] End Time: 완전 충진까지 자동 설정, 최대 3분(180초) ──
    etime = st.number_input(
        "End Time (s)",
        min_value=0.1, max_value=180.0,
        value=min(float(st.session_state["etime"]), 180.0),
        step=0.1, key="etime",
        help=(
            "이론 충진 시간 × 1.2 여유를 자동 설정합니다.\n"
            "수동 입력도 가능. 최대 3분(180초) 제한.\n"
            "충진 미완료 시 이 값을 늘려주세요."
        )
    )

    if st.button("✅ Confirm Process", use_container_width=True):
        st.session_state["process_confirmed"] = True
        st.toast("Process confirmed!", icon="✅")

    st.divider()
    num_frames_sel = st.select_slider("Animation Frames", options=[5, 10, 15, 20, 30], value=st.session_state.get("num_frames", 15))
    st.session_state["num_frames"] = num_frames_sel

    run_disabled = (
        st.session_state["sim_running"]
        or not st.session_state["props_confirmed"]
        or not st.session_state.get("process_confirmed")
    )

    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True, disabled=run_disabled):
        if not ZAPIER_URL:
            st.error("ZAPIER_URL not configured")
        else:
            clear_old_results()
            sig_id  = str(uuid.uuid4())[:8]
            res_mm  = compute_voxel_res_mm(mesh_obj) if mesh_obj else 0.5

            st.session_state["executed_params"] = {
                "signal_id":   sig_id, "material": st.session_state["mat_name"],
                "viscosity":   float(st.session_state["props"]["nu"]),
                "density":     float(st.session_state["props"]["rho"]),
                "melt_temp":   float(st.session_state["props"]["Tmelt"]),
                "temp":        float(temp_c),
                "press":       float(press_mpa), "vel_mms": float(vel_mms),
                "etime":       float(etime),
                "gate_x": gx, "gate_y": gy, "gate_z": gz, "gate_dia": float(g_size),
                "num_frames":  num_frames_sel, "mesh_res_mm": res_mm,
            }
            st.session_state.update({
                "last_signal_id": sig_id, "sim_running": True,
                "sim_status": "running", "voxel_cache": None
            })
            add_log(f"🚀 Launched | End Time: {etime:.1f}s | Gate: ({gx:.2f},{gy:.2f},{gz:.2f})")

            payload = {
                "signal_id": sig_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material":  st.session_state["executed_params"]["material"],
                "viscosity": st.session_state["executed_params"]["viscosity"],
                "density":   st.session_state["executed_params"]["density"],
                "temp":      st.session_state["executed_params"]["temp"],
                "press":     st.session_state["executed_params"]["press"],
                "vel":       round(vel_mms / 1000, 6),
                "etime":     float(etime),
                "num_frames": num_frames_sel,
                "mesh_resolution": res_mm / 1000.0,
                "gate_pos":  {"x": round(gx,3), "y": round(gy,3), "z": round(gz,3)},
                "gate_size": float(g_size),
            }
            try:
                r = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if r.status_code == 200:
                    st.toast(f"Signal sent! ID: {sig_id}", icon="🚀")
                else:
                    st.error(f"Failed: HTTP {r.status_code}")
                    st.session_state["sim_running"] = False
            except Exception as e:
                st.error(f"Error: {e}"); st.session_state["sim_running"] = False

# ═══════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════
mesh_obj = st.session_state.get("mesh")

col_geo, col_log = st.columns([2, 1])
with col_geo:
    st.header("🎥 3D Geometry & Gate")
    if mesh_obj:
        v, f = mesh_obj.vertices, mesh_obj.faces

        # AI 게이트 후보들도 시각화
        fig = go.Figure(data=[
            go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2], i=f[:,0],j=f[:,1],k=f[:,2],
                      color="#AAAAAA", opacity=0.7),
            go.Scatter3d(
                x=[st.session_state["gx_final"]],
                y=[st.session_state["gy_final"]],
                z=[st.session_state["gz_final"]],
                mode="markers",
                marker=dict(size=st.session_state["gsize"]*3, color="red"),
                name="Gate (Selected)"
            )
        ])

        # AI 후보 위치도 표시
        gate_suggestions = st.session_state.get("gate_suggestions", [])
        if gate_suggestions:
            sx = [s["pos"][0] for s in gate_suggestions]
            sy = [s["pos"][1] for s in gate_suggestions]
            sz = [s["pos"][2] for s in gate_suggestions]
            labels = [s["label"] for s in gate_suggestions]
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode="markers+text",
                text=labels, textposition="top center",
                marker=dict(size=6, color="orange", symbol="diamond"),
                name="AI Gate Candidates"
            ))

        fig.update_layout(
            margin=dict(l=0,r=0,b=0,t=0),
            scene=dict(aspectmode="data"),
            height=500,
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})
    else:
        st.info("Upload STL file to see 3D model")

with col_log:
    st.header("📟 Simulation Logs")
    lc = st.container(height=350)
    with lc:
        for log in st.session_state["sim_logs"][-25:]: st.code(log, language="bash")
    if st.button("🗑 Clear Logs", use_container_width=True):
        st.session_state["sim_logs"] = []; st.rerun()

st.title("📊 Simulation Results")
cr1, cr2 = st.columns([2, 1])
with cr1: st.markdown("### Download & Sync")
with cr2:
    if st.button("🔄 Sync Results", use_container_width=True, type="primary"):
        if sync_simulation_results(): st.rerun()

summary_text = build_summary_text()
if summary_text: st.text_area("📄 Simulation Summary", summary_text, height=230)

# ═══════════════════════════════════════════════════════════════
#  ★★★ 3D FILLING ANIMATION – SOLID VOXEL ENGINE ★★★
# ═══════════════════════════════════════════════════════════════
vtk_dir = "VTK"
if os.path.exists(vtk_dir) and mesh_obj is not None:
    st.subheader("🌊 3D Filling Animation (Forced 100% Visual Flow)")

    num_frames    = st.session_state.get("num_frames", 15)
    sampled_files = sample_vtk_files(vtk_dir, num_frames)
    total_steps   = len(sampled_files)

    if total_steps > 0:
        with st.expander("🔧 Visualization Settings", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mold_opacity  = st.slider("Mold Opacity",  0.0, 0.5, 0.10, 0.01)
                fluid_opacity = st.slider("Fluid Opacity", 0.3, 1.0, 0.90, 0.05)
            with c2:
                default_res = compute_voxel_res_mm(mesh_obj)
                res_mm_ui = st.slider("Voxel Resolution (mm)", 0.15, 2.0,
                                      float(round(default_res, 2)), 0.05,
                                      help="v2.7: 기본값이 기존 1/5 → 1/10으로 세밀화됨")
                cache = st.session_state.get("voxel_cache")
                if cache and abs(cache["res_mm"] - res_mm_ui) > 0.05:
                    st.session_state["voxel_cache"] = None
            with c3:
                max_vox_render = st.select_slider("Max Voxels (render)",
                                                  options=[2000, 4000, 6000, 8000, 12000], value=6000)

        # ─── Animation Controls ───
        cc1, cc2, cc3, cc4 = st.columns([1, 1, 3, 1])
        with cc1:
            if st.button("⏮ First", use_container_width=True):
                st.session_state.update({"current_frame":0, "animation_playing":False}); st.rerun()
        with cc2:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state["animation_playing"] = False; st.rerun()
        with cc3:
            if st.button("▶ Play", use_container_width=True, type="primary"):
                st.session_state["animation_playing"] = True; st.rerun()
        with cc4:
            if st.button("⏭ Last", use_container_width=True):
                st.session_state.update({"current_frame": total_steps-1, "animation_playing":False}); st.rerun()

        current_frame = st.slider("Frame", 0, total_steps - 1,
                                  value=min(st.session_state.get("current_frame", 0), total_steps-1))
        st.session_state["current_frame"] = current_frame

        gate_mm = (st.session_state["gx_final"], st.session_state["gy_final"], st.session_state["gz_final"])

        with st.spinner("⚙️ Calculating complete flow paths..."):
            cache = get_or_build_voxel_cache(mesh_obj, gate_mm, res_mm_ui)

        if cache and cache["total"] > 0:
            fill_order = cache["fill_order"]
            total_vox  = cache["total"]
            origin     = cache["origin"]

            # ─── 강제 100% 시각화 (프레임 기반) ───
            visual_ratio = (current_frame + 1) / total_steps
            n_show = max(1, int(visual_ratio * total_vox))

            fpath     = sampled_files[current_frame]
            vtk_ratio = read_alpha_fill_ratio(fpath)

            if vtk_ratio is not None:
                ratio_msg = f"Visual: {visual_ratio*100:.1f}% (Actual VTK: {vtk_ratio*100:.1f}%)"
            else:
                ratio_msg = f"Visual: {visual_ratio*100:.1f}%"

            shown_voxels = fill_order[:n_show]
            fill_vals    = np.linspace(0.0, 1.0, n_show)

            pts, fi, fj, fk, intensity = voxels_to_mesh3d(
                shown_voxels, origin, res_mm_ui, fill_vals, max_vox=max_vox_render)

            fig = go.Figure()
            fig.add_trace(make_mold_trace(mesh_obj, opacity=mold_opacity))
            fig.add_trace(go.Scatter3d(
                x=[st.session_state["gx_final"]],
                y=[st.session_state["gy_final"]],
                z=[st.session_state["gz_final"]],
                mode="markers",
                marker=dict(size=st.session_state["gsize"]*2, color="red",
                            symbol="x", line=dict(width=2, color="white")),
                name="Gate"
            ))

            if pts is not None:
                fig.add_trace(make_solid_fluid_trace(pts, fi, fj, fk, intensity, fluid_opacity))
                if vtk_ratio is not None and vtk_ratio < 0.95 and current_frame == total_steps - 1:
                    st.warning(
                        f"⚠️ 실제 서버 결과(Actual VTK)는 {vtk_ratio*100:.1f}% 에서 멈춘 Short Shot 상태입니다. "
                        f"(End Time을 확인하세요) 화면은 강제로 100%를 시각화했습니다."
                    )
                else:
                    st.success(f"Frame {current_frame+1}/{total_steps} | Voxels: {n_show:,}/{total_vox:,} | {ratio_msg}")

            fig.update_layout(
                scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                height=650, margin=dict(l=0,r=0,b=0,t=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.get("animation_playing", False):
            time.sleep(0.3)
            st.session_state["current_frame"] = (current_frame + 1) % total_steps
            st.rerun()

elif os.path.exists(vtk_dir) and mesh_obj is None:
    st.warning("⚠️ Upload STL file to enable 3D solid voxel animation.")

# ═══════════════════════════════════════════════════════════════
#  재료 DB 관리 페이지 (사이드바 하단 숨김 → 메인 영역 확장 섹션)
# ═══════════════════════════════════════════════════════════════
with st.expander("🗂️ Material DB 관리 (material_property.txt)", expanded=False):
    st.caption(f"파일 경로: `{MATERIAL_FILE}`")
    db_now = load_material_db(MATERIAL_FILE)
    if db_now:
        import pandas as pd
        df = pd.DataFrame([
            {"Material": k, "nu (m²/s)": f"{v['nu']:.2e}", "rho (kg/m³)": v["rho"],
             "Tmelt (°C)": v["Tmelt"], "Tmold (°C)": v["Tmold"],
             "Press (MPa)": v["press_mpa"], "Vel (mm/s)": v["vel_mms"]}
            for k, v in sorted(db_now.items())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("DB가 비어 있습니다. material_property.txt 파일을 확인하세요.")

    st.markdown("**새 재료 추가 / 기존 재료 수정**")
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        new_mat   = st.text_input("재료명", key="new_mat_name")
        new_nu    = st.number_input("Viscosity (m²/s)", value=4e-3, format="%.2e", key="new_nu")
    with nc2:
        new_rho   = st.number_input("Density (kg/m³)",  value=7800.0, key="new_rho")
        new_tmelt = st.number_input("Melt Temp (°C)",    value=185.0,  key="new_tmelt")
    with nc3:
        new_tmold = st.number_input("Mold Temp (°C)",    value=40.0,   key="new_tmold")
        new_press = st.number_input("Press (MPa)",        value=110.0,  key="new_press")
        new_vel   = st.number_input("Vel (mm/s)",         value=25.0,   key="new_vel")

    if st.button("💾 DB에 추가/갱신", type="primary", use_container_width=True):
        if new_mat.strip():
            ok = save_material_to_txt(new_mat.strip(), {
                "nu": new_nu, "rho": new_rho, "Tmelt": new_tmelt,
                "Tmold": new_tmold, "press_mpa": new_press, "vel_mms": new_vel
            })
            if ok:
                st.success(f"✅ '{new_mat.upper()}' → material_property.txt 저장 완료")
                st.rerun()
        else:
            st.warning("재료명을 입력하세요.")
