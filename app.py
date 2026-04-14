"""
MIM-Ops Pro v2.4
=================
핵심 변경:
  - 몰드 STL을 내부 voxel grid로 변환 (최솟 두께 / 5 해상도)
  - 게이트 voxel에서 BFS로 3D 확산 순서 계산
  - VTK alpha 필드로 총 충진 비율 계산 (없으면 프레임 번호 비례)
  - Plotly Mesh3d로 solid cube (8꼭짓점 × 6면) 렌더링 → 진짜 3D solid fill
  - clip_surface 완전 제거, extract_surface 제거
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

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# ───────────────────── Session State ─────────────────────
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0);  _init("gy", 0.0);  _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp", 230.0); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 0.5)
_init("sim_running", False); _init("sim_status", "idle")
_init("sim_logs", [])
_init("last_signal_id", None)
_init("mesh", None)
_init("props", None); _init("props_confirmed", False)
_init("process_confirmed", False)
_init("mat_name", "PA66+30GF")
_init("last_vel_mms", 80.0); _init("last_etime", 0.5)
_init("gx_final", 0.0); _init("gy_final", 0.0); _init("gz_final", 0.0)
_init("animation_playing", False); _init("current_frame", 0)
_init("vtk_files", [])
_init("last_synced_signal_id", None)
_init("executed_params", None)
_init("num_frames", 15)
# voxel cache – rebuilt only when mesh/resolution/gate changes
_init("voxel_cache", None)   # dict: {occupied, bfs_order, res_mm, gate, origin, total}

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

# ───────────────────── VTK helpers ─────────────────────
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
    """
    VTK 파일에서 alpha field의 충진 비율(0~1)을 반환.
    = alpha > 0.05 인 셀 비율
    """
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
#  ★★★ SOLID VOXEL ENGINE ★★★
# ═══════════════════════════════════════════════════════════

def compute_voxel_res_mm(mold_trimesh):
    """최솟 두께 / 5 (범위: 0.3 ~ 2.0 mm)"""
    bb   = mold_trimesh.bounds
    dims = np.array(bb[1]) - np.array(bb[0])
    valid = dims[dims > 0.1]
    min_dim = float(np.min(valid)) if len(valid) else 10.0
    return float(np.clip(min_dim / 5.0, 0.3, 2.0))

def build_voxel_grid(mold_trimesh, res_mm):
    """
    trimesh → 내부 voxel bool grid.
    Returns: occupied(bool 3D), origin(mm, 3), res_mm
    """
    vox = mold_trimesh.voxelized(pitch=res_mm)
    vox = vox.fill()
    occupied = vox.matrix.copy()              # bool (Nx, Ny, Nz)
    origin   = np.array(vox.translation, dtype=float)
    return occupied, origin

def gate_to_voxel(gate_mm, origin, res_mm, shape):
    """게이트 좌표(mm) → voxel index, 격자 안으로 클리핑"""
    idx = np.round((np.array(gate_mm) - origin) / res_mm).astype(int)
    idx = np.clip(idx, 0, np.array(shape) - 1)
    return tuple(idx)

def bfs_fill_order(occupied, start_vox):
    """
    26-connectivity BFS.
    시작점이 빈 셀이면 가장 가까운 내부 셀로 보정.
    Returns list of (ix, iy, iz) in fill order.
    """
    Nx, Ny, Nz = occupied.shape
    sx, sy, sz = int(start_vox[0]), int(start_vox[1]), int(start_vox[2])

    if not occupied[sx, sy, sz]:
        occ_idx = np.argwhere(occupied)
        if len(occ_idx) == 0:
            return []
        dists = np.sum((occ_idx - np.array([sx, sy, sz]))**2, axis=1)
        nearest = occ_idx[np.argmin(dists)]
        sx, sy, sz = int(nearest[0]), int(nearest[1]), int(nearest[2])

    visited = np.zeros_like(occupied, dtype=bool)
    visited[sx, sy, sz] = True
    queue  = deque([(sx, sy, sz)])
    order  = []

    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]

    while queue:
        cx, cy, cz = queue.popleft()
        order.append((cx, cy, cz))
        for dx, dy, dz in offsets:
            nx2, ny2, nz2 = cx+dx, cy+dy, cz+dz
            if (0 <= nx2 < Nx and 0 <= ny2 < Ny and 0 <= nz2 < Nz
                    and occupied[nx2, ny2, nz2]
                    and not visited[nx2, ny2, nz2]):
                visited[nx2, ny2, nz2] = True
                queue.append((nx2, ny2, nz2))

    return order

def get_or_build_voxel_cache(mold_trimesh, gate_mm, res_mm):
    """캐시 유효성 검사 → 필요하면 재빌드"""
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
    add_log(f"Grid {shape}, gate voxel {start_vox}, occupied={int(occupied.sum()):,}")
    bfs_order = bfs_fill_order(occupied, start_vox)
    add_log(f"BFS done: {len(bfs_order):,} reachable voxels")

    cache = {
        "occupied":  occupied,
        "origin":    origin,
        "res_mm":    res_mm,
        "gate":      gate_key,
        "bfs_order": bfs_order,
        "total":     len(bfs_order),
    }
    st.session_state["voxel_cache"] = cache
    return cache

def voxels_to_mesh3d(vox_indices, origin, res_mm, fill_ratios=None, max_vox=6000):
    """
    voxel list → Plotly Mesh3d (solid cubes).
    각 voxel = 8 vertices + 12 triangles.
    max_vox : 렌더링 최대 voxel 수 (성능 제한)
    """
    vox_arr = np.array(vox_indices, dtype=float)   # (N, 3)
    N = len(vox_arr)
    if N == 0:
        return None, None, None, None, None

    # 균등 decimation
    if N > max_vox:
        idx = np.round(np.linspace(0, N-1, max_vox)).astype(int)
        vox_arr = vox_arr[idx]
        if fill_ratios is not None:
            fill_ratios = np.asarray(fill_ratios)[idx]
        N = max_vox

    h = res_mm * 0.5  # voxel half-size

    # 8 corner offsets of a unit cube
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                         [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]],
                        dtype=float) * h  # (8,3)

    # 12 triangles (2 per face × 6 faces)
    tri = np.array([[0,1,2],[0,2,3],   # bottom
                    [4,6,5],[4,7,6],   # top
                    [0,4,5],[0,5,1],   # front
                    [2,6,7],[2,7,3],   # back
                    [0,3,7],[0,7,4],   # left
                    [1,5,6],[1,6,2]])  # right  (12,3)

    # 모든 voxel의 center (mm)
    centers = origin + (vox_arr + 0.5) * res_mm   # (N,3)

    # pre-allocate
    all_pts = np.empty((N * 8, 3), dtype=float)
    all_i   = np.empty(N * 12, dtype=np.int32)
    all_j   = np.empty(N * 12, dtype=np.int32)
    all_k   = np.empty(N * 12, dtype=np.int32)
    all_int = np.empty(N * 8,  dtype=float)

    for ci in range(N):
        pts      = corners + centers[ci]            # (8,3)
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
        colorbar=dict(title="Fill Order\n(Blue=Gate → Red=Last)", thickness=15, len=0.6)
    )

# ───────────────────── Summary ─────────────────────
def build_summary_text():
    ep = st.session_state.get("executed_params")
    if ep is None:
        return None
    lines = [
        "Simulation Status: Success",
        f"Material: {ep.get('material','N/A')}",
        f"Velocity: {ep.get('vel_mms',0)/1000:.4f} m/s   ({ep.get('vel_mms',0):.1f} mm/s)",
        f"Viscosity: {ep.get('viscosity',0):.2e} m²/s",
        f"Density: {ep.get('density',0):.0f} kg/m³",
        f"Melt Temp: {ep.get('melt_temp',0):.1f} °C",
        f"Mold Temp: {ep.get('mold_temp',0):.1f} °C",
        f"Injection Temp: {ep.get('temp',0):.1f} °C",
        f"Pressure: {ep.get('press',0):.1f} MPa",
        f"End Time: {ep.get('etime',0):.2f} s",
        f"Gate: ({ep.get('gate_x',0):.2f}, {ep.get('gate_y',0):.2f}, {ep.get('gate_z',0):.2f}) mm",
        f"Gate Dia: {ep.get('gate_dia',0):.1f} mm",
        f"Signal ID: {ep.get('signal_id','N/A')}",
    ]
    if os.path.exists("results.txt"):
        with open("results.txt") as f:
            raw = f.read()
        for kw in ["Last Time Step","Time Steps","Finish Time"]:
            m = re.search(rf"{kw}[:\s]+(.+)", raw)
            if m:
                lines.append(f"{kw}: {m.group(1).strip()}")
    return "\n".join(lines)

# ───────────────────── Material DB ─────────────────────
LOCAL_DB = {
    "PP":        {"nu":1e-3,  "rho":900.0,  "Tmelt":230.0,"Tmold":40.0, "press_mpa":70.0, "vel_mms":80.0, "desc":"General-purpose polypropylene"},
    "ABS":       {"nu":2e-3,  "rho":1050.0, "Tmelt":240.0,"Tmold":60.0, "press_mpa":80.0, "vel_mms":70.0, "desc":"ABS resin"},
    "PA66":      {"nu":5e-4,  "rho":1140.0, "Tmelt":280.0,"Tmold":80.0, "press_mpa":90.0, "vel_mms":100.0,"desc":"Nylon 66"},
    "PA66+30GF": {"nu":4e-4,  "rho":1300.0, "Tmelt":285.0,"Tmold":85.0, "press_mpa":110.0,"vel_mms":80.0, "desc":"30% glass-fiber reinforced nylon"},
    "PC":        {"nu":3e-3,  "rho":1200.0, "Tmelt":300.0,"Tmold":85.0, "press_mpa":120.0,"vel_mms":60.0, "desc":"Polycarbonate"},
    "POM":       {"nu":8e-4,  "rho":1410.0, "Tmelt":200.0,"Tmold":90.0, "press_mpa":85.0, "vel_mms":90.0, "desc":"Polyacetal"},
    "HDPE":      {"nu":9e-4,  "rho":960.0,  "Tmelt":220.0,"Tmold":35.0, "press_mpa":60.0, "vel_mms":90.0, "desc":"HDPE"},
    "PET":       {"nu":6e-4,  "rho":1370.0, "Tmelt":265.0,"Tmold":70.0, "press_mpa":80.0, "vel_mms":85.0, "desc":"PET"},
    "CATAMOLD":  {"nu":5e-3,  "rho":4900.0, "Tmelt":185.0,"Tmold":40.0, "press_mpa":100.0,"vel_mms":30.0, "desc":"BASF Catamold MIM feedstock"},
    "MIM":       {"nu":5e-3,  "rho":5000.0, "Tmelt":185.0,"Tmold":40.0, "press_mpa":100.0,"vel_mms":30.0, "desc":"Metal injection molding feedstock"},
    "17-4PH":    {"nu":4e-3,  "rho":7780.0, "Tmelt":185.0,"Tmold":40.0, "press_mpa":110.0,"vel_mms":25.0, "desc":"17-4PH stainless steel"},
    "316L":      {"nu":4e-3,  "rho":7900.0, "Tmelt":185.0,"Tmold":40.0, "press_mpa":110.0,"vel_mms":25.0, "desc":"316L stainless steel"},
}

def get_props(material):
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Database"}
    return {"nu":1e-3,"rho":1000.0,"Tmelt":220.0,"Tmold":50.0,
            "press_mpa":70.0,"vel_mms":80.0,"material":material,
            "source":"Default","desc":f"{material} — default values"}

def get_process(material):
    p = get_props(material)
    return {"temp":float(p["Tmelt"]),"press":float(p["press_mpa"]),"vel":float(p["vel_mms"])}

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded:
        try:
            mesh_obj = trimesh.load(uploaded, file_type="stl")
            st.session_state["mesh"]        = mesh_obj
            st.session_state["voxel_cache"] = None   # 새 STL → 캐시 무효화
            st.success(f"✅ STL loaded — {len(mesh_obj.faces):,} faces")
            add_log(f"STL loaded: {len(mesh_obj.faces):,} faces")
        except Exception as e:
            st.error(f"STL load failed: {e}")

    st.divider()
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        m = st.session_state.get("mesh")
        if m:
            center = m.centroid
            snap, _, _ = trimesh.proximity.closest_point(m, [center])
            pos = snap[0]
            st.session_state.update({
                "gx": float(pos[0]), "gy": float(pos[1]),
                "gz": float(pos[2]), "gsize": 2.5,
                "voxel_cache": None
            })
            st.toast("Gate suggested!", icon="🪄")
            add_log(f"Gate: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        else:
            st.warning("Upload STL first")

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")
    vx = st.number_input("Gate X", value=float(st.session_state["gx"]), step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=float(st.session_state["gy"]), step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=float(st.session_state["gz"]), step=0.1, key="gz")

    mesh_obj = st.session_state.get("mesh")
    if mesh_obj:
        snap, _, _ = trimesh.proximity.closest_point(mesh_obj, [[vx, vy, vz]])
        gx, gy, gz = float(snap[0][0]), float(snap[0][1]), float(snap[0][2])
    else:
        gx, gy, gz = vx, vy, vz
    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz

    st.divider()
    st.header("🧪 3. Material")
    mat_name = st.text_input("Material Name", value=st.session_state["mat_name"], key="mat_name_input")
    st.session_state["mat_name"] = mat_name
    if st.button("🤖 AI Material Properties", use_container_width=True, type="primary"):
        st.session_state["props"]           = get_props(mat_name)
        st.session_state["props_confirmed"] = False
        add_log(f"Material properties loaded: {mat_name}")

    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 Source: {p.get('source','Database')}")
        if p.get("desc"):
            st.info(p["desc"])
        with st.expander("📋 Edit Properties", expanded=True):
            p["nu"]    = st.number_input("Viscosity (m²/s)", value=float(p["nu"]),    format="%.2e", key="edit_nu")
            p["rho"]   = st.number_input("Density (kg/m³)",  value=float(p["rho"]),              key="edit_rho")
            p["Tmelt"] = st.number_input("Melt Temp (°C)",   value=float(p["Tmelt"]),             key="edit_tmelt")
            p["Tmold"] = st.number_input("Mold Temp (°C)",   value=float(p["Tmold"]),             key="edit_tmold")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Confirm\nProperties", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Properties confirmed!", icon="✅")
                    add_log("Material properties confirmed")
            with c2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state["props"]           = None
                    st.session_state["props_confirmed"] = False
                    add_log("Properties reset"); st.rerun()

    st.divider()
    st.header("⚙️ 4. Process")
    if st.button("🤖 Optimize Process", use_container_width=True):
        opt = get_process(mat_name)
        st.session_state.update({"temp": opt["temp"], "press": opt["press"], "vel": opt["vel"]})
        st.toast("Process optimized!", icon="🤖")
    temp_c    = st.number_input("Temp (°C)",       50.0,  450.0, value=float(st.session_state["temp"]),  step=1.0, key="temp")
    press_mpa = st.number_input("Pressure (MPa)",  10.0,  250.0, value=float(st.session_state["press"]), step=1.0, key="press")
    vel_mms   = st.number_input("Velocity (mm/s)",  1.0,  600.0, value=float(st.session_state["vel"]),   step=1.0, key="vel")
    etime     = st.number_input("End Time (s)", value=float(st.session_state["etime"]),
                                min_value=0.1, max_value=10.0, step=0.1, key="etime")
    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"]   = etime

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Confirm Process", use_container_width=True):
            st.session_state["process_confirmed"] = True
            st.toast("Process confirmed!", icon="✅")
            add_log("Process conditions confirmed")
    with c2:
        if st.button("🔄 Reset Process", use_container_width=True):
            st.session_state["process_confirmed"] = False
            add_log("Process reset"); st.rerun()
    if not st.session_state.get("process_confirmed", False):
        st.warning("⚠️ Please confirm process conditions")

    st.divider()
    num_frames_sel = st.select_slider("Animation Frames",
                                      options=[5, 10, 15, 20, 30],
                                      value=st.session_state.get("num_frames", 15))
    st.session_state["num_frames"] = num_frames_sel

    run_disabled = (st.session_state["sim_running"]
                    or not st.session_state["props_confirmed"]
                    or not st.session_state.get("process_confirmed", False))
    if st.button("🚀 Run Cloud Simulation", type="primary",
                 use_container_width=True, disabled=run_disabled):
        if not ZAPIER_URL:
            st.error("ZAPIER_URL not configured")
        else:
            clear_old_results()
            sig_id  = str(uuid.uuid4())[:8]
            mold_tm = st.session_state.get("mesh")
            res_mm  = compute_voxel_res_mm(mold_tm) if mold_tm else 1.0

            st.session_state["executed_params"] = {
                "signal_id":   sig_id,
                "material":    st.session_state["mat_name"],
                "viscosity":   float(st.session_state["props"]["nu"]),
                "density":     float(st.session_state["props"]["rho"]),
                "melt_temp":   float(st.session_state["props"]["Tmelt"]),
                "mold_temp":   float(st.session_state["props"]["Tmold"]),
                "temp":        float(temp_c),
                "press":       float(press_mpa),
                "vel_mms":     float(vel_mms),
                "etime":       float(etime),
                "gate_x":      gx, "gate_y": gy, "gate_z": gz,
                "gate_dia":    float(g_size),
                "num_frames":  num_frames_sel,
                "mesh_res_mm": res_mm,
            }
            st.session_state.update({
                "last_signal_id": sig_id,
                "sim_running":    True,
                "sim_status":     "running",
                "voxel_cache":    None,
            })
            add_log(f"🚀 Launched | ID: {sig_id}")
            add_log(f"Material: {st.session_state['executed_params']['material']}")
            add_log(f"Gate: ({gx:.2f}, {gy:.2f}, {gz:.2f}) mm, Dia={g_size}mm")
            add_log(f"Mesh res: {res_mm:.2f} mm (min_dim/5)")
            add_log(f"Injection: {temp_c}°C, {press_mpa}MPa, {vel_mms}mm/s")

            payload = {
                "signal_id":       sig_id,
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material":        st.session_state["executed_params"]["material"],
                "viscosity":       st.session_state["executed_params"]["viscosity"],
                "density":         st.session_state["executed_params"]["density"],
                "melt_temp":       st.session_state["executed_params"]["melt_temp"],
                "mold_temp":       st.session_state["executed_params"]["mold_temp"],
                "temp":            st.session_state["executed_params"]["temp"],
                "press":           st.session_state["executed_params"]["press"],
                "vel":             round(vel_mms / 1000, 6),
                "etime":           float(etime),
                "num_frames":      num_frames_sel,
                "mesh_resolution": res_mm / 1000.0,
                "gate_pos":        {"x":round(gx,3),"y":round(gy,3),"z":round(gz,3)},
                "gate_size":       float(g_size),
            }
            try:
                add_log("Sending to Zapier...")
                r = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if r.status_code == 200:
                    st.toast(f"Signal sent! ID: {sig_id}", icon="🚀")
                    add_log(f"Sent (HTTP {r.status_code})")
                    add_log("⏳ Wait 2-3 min then click 'Sync Results'.")
                else:
                    st.error(f"Failed: HTTP {r.status_code}")
                    st.session_state["sim_running"] = False
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state["sim_running"] = False

# ═══════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════
col_geo, col_log = st.columns([2, 1])
with col_geo:
    st.header("🎥 3D Geometry & Gate")
    mesh_obj = st.session_state.get("mesh")
    if mesh_obj:
        v, f = mesh_obj.vertices, mesh_obj.faces
        fig = go.Figure(data=[
            go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],
                      i=f[:,0],j=f[:,1],k=f[:,2],
                      color="#AAAAAA", opacity=0.7),
            go.Scatter3d(x=[st.session_state["gx_final"]],
                         y=[st.session_state["gy_final"]],
                         z=[st.session_state["gz_final"]],
                         mode="markers",
                         marker=dict(size=st.session_state["gsize"]*3, color="red"),
                         name="Gate")
        ])
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),
                          scene=dict(aspectmode="data"), height=500)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})
        bb = mesh_obj.bounds
        c1,c2,c3 = st.columns(3)
        c1.metric("X Size", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y Size", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z Size", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("Upload STL file to see 3D model")

with col_log:
    st.header("📟 Simulation Logs")
    if st.session_state["sim_running"]:
        st.info("🟢 STATUS: RUNNING...")
    elif st.session_state["sim_status"] == "failed":
        st.error("🔴 STATUS: FAILED")
    else:
        st.success("✅ STATUS: READY")
    lc = st.container(height=350)
    with lc:
        for log in st.session_state["sim_logs"][-25:]:
            st.code(log, language="bash")
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state["sim_running"]:
            if st.button("✅ Mark Complete", use_container_width=True):
                st.session_state["sim_running"] = False
                st.session_state["sim_status"]  = "completed"
                add_log("Marked as completed"); st.rerun()
    with c2:
        if st.button("🗑 Clear Logs", use_container_width=True):
            st.session_state["sim_logs"] = []; st.rerun()

st.info(f"📍 Final Gate: ({st.session_state['gx_final']:.2f}, {st.session_state['gy_final']:.2f}, {st.session_state['gz_final']:.2f}) mm")
if st.session_state["props_confirmed"] and st.session_state.get("process_confirmed") and st.session_state["props"]:
    p = st.session_state["props"]
    st.caption(f"ℹ️ {st.session_state['mat_name']} | nu={p['nu']:.2e} | rho={p['rho']} kg/m³ | Tmelt={p['Tmelt']}°C")
else:
    st.caption("ℹ️ Confirm properties and process before running simulation.")

# ─── Results ───
st.title("📊 Simulation Results")
cr1, cr2, cr3 = st.columns([2, 1, 1])
with cr1:
    st.markdown("### Download & Sync")
with cr2:
    if st.button("🔄 Sync Results", use_container_width=True, type="primary"):
        if sync_simulation_results(): st.rerun()
with cr3:
    if st.button("🗑 Clear Results", use_container_width=True):
        clear_old_results(); st.success("Results cleared"); st.rerun()

summary_text = build_summary_text()
if summary_text:
    st.text_area("📄 Simulation Summary", summary_text, height=230)
    ep = st.session_state.get("executed_params")
    sid = st.session_state.get("last_synced_signal_id")
    if sid and ep:
        if sid != ep["signal_id"]:
            st.warning(f"⚠️ Synced: older run (ID: {sid}). Current: {ep['signal_id']}")
        else:
            st.success("Results synchronized successfully!")
else:
    st.info("No results loaded. Run a simulation and click 'Sync Results'.")

if os.path.exists("logs.zip"):
    with open("logs.zip","rb") as f:
        st.download_button("📂 Download Logs", f, "logs.zip", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  ★★★ 3D FILLING ANIMATION – SOLID VOXEL ENGINE ★★★
# ═══════════════════════════════════════════════════════════════
vtk_dir      = "VTK"
mold_trimesh = st.session_state.get("mesh")

if os.path.exists(vtk_dir) and mold_trimesh is not None:
    st.subheader("🌊 3D Filling Animation (Solid Voxel Mesh)")

    num_frames    = st.session_state.get("num_frames", 15)
    sampled_files = sample_vtk_files(vtk_dir, num_frames)
    total_steps   = len(sampled_files)

    if total_steps == 0:
        st.warning("No VTK files found. Please sync results first.")
    else:
        with st.expander("🔧 Visualization Settings", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mold_opacity  = st.slider("Mold Opacity",  0.0, 0.5, 0.10, 0.01)
                fluid_opacity = st.slider("Fluid Opacity", 0.3, 1.0, 0.90, 0.05)
            with c2:
                default_res = compute_voxel_res_mm(mold_trimesh)
                res_mm_ui = st.slider(
                    "Voxel Resolution (mm)",
                    min_value=0.3, max_value=3.0,
                    value=float(round(default_res, 1)), step=0.1,
                    help="기본값 = 최솟 두께 / 5. 작을수록 정밀하지만 느림."
                )
                cache = st.session_state.get("voxel_cache")
                if cache and abs(cache["res_mm"] - res_mm_ui) > 0.05:
                    st.session_state["voxel_cache"] = None
            with c3:
                view_mode = st.radio("View Mode", ["Auto","Uniform"], index=0)
                max_vox_render = st.select_slider(
                    "Max Voxels (render)",
                    options=[2000, 4000, 6000, 8000, 12000],
                    value=6000,
                    help="많을수록 정밀하나 느림"
                )

        # ─── Animation Controls ───
        st.markdown("### 🎮 Animation Controls")
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
                st.session_state.update({"current_frame":total_steps-1,"animation_playing":False}); st.rerun()

        current_frame = st.slider(
            "Frame", 0, total_steps - 1,
            value=min(st.session_state.get("current_frame", 0), total_steps-1),
            key="frame_slider"
        )
        st.session_state["current_frame"] = current_frame

        # ─── Voxel grid & BFS (캐시) ───
        gate_mm = (st.session_state["gx_final"],
                   st.session_state["gy_final"],
                   st.session_state["gz_final"])

        with st.spinner("⚙️ Building 3D voxel grid inside mold (cached after first build)…"):
            try:
                cache = get_or_build_voxel_cache(mold_trimesh, gate_mm, res_mm_ui)
            except Exception as e:
                st.error(f"Voxel build error: {e}")
                cache = None

        if cache is None or cache["total"] == 0:
            st.warning("Voxel grid is empty. Check STL mesh or gate position.")
        else:
            bfs_order = cache["bfs_order"]
            total_vox = cache["total"]
            origin    = cache["origin"]

            # ─── 충진 비율 결정 ───
            fpath     = sampled_files[current_frame]
            vtk_ratio = read_alpha_fill_ratio(fpath)
            if vtk_ratio is not None:
                fill_ratio = vtk_ratio
                ratio_src  = f"VTK alpha={fill_ratio:.3f}"
            else:
                fill_ratio = (current_frame + 1) / total_steps
                ratio_src  = f"frame {current_frame+1}/{total_steps}"

            n_show       = max(1, int(fill_ratio * total_vox))
            shown_voxels = bfs_order[:n_show]
            fill_vals    = np.linspace(0.0, 1.0, n_show)   # 게이트=파랑, 끝=빨강

            with st.spinner(f"Rendering {n_show:,} solid voxels…"):
                pts, fi, fj, fk, intensity = voxels_to_mesh3d(
                    shown_voxels, origin, res_mm_ui, fill_vals, max_vox=max_vox_render
                )

            fig = go.Figure()
            fig.add_trace(make_mold_trace(mold_trimesh, opacity=mold_opacity))
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
                st.success(
                    f"Frame {current_frame+1}/{total_steps} | "
                    f"Voxels: {n_show:,}/{total_vox:,} ({fill_ratio*100:.1f}%) | "
                    f"Res: {res_mm_ui:.2f} mm | Source: {ratio_src}"
                )
            else:
                st.info("No voxels rendered. Check gate position and STL.")

            fig.update_layout(
                scene=dict(
                    aspectmode="data" if view_mode=="Auto" else "cube",
                    camera=dict(eye=dict(x=1.5,y=1.5,z=1.5)),
                    xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"
                ),
                height=650, margin=dict(l=0,r=0,b=0,t=0)
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom":True,"displayModeBar":True})

        if st.session_state.get("animation_playing", False):
            nxt = (current_frame + 1) % total_steps
            st.session_state["current_frame"] = nxt
            time.sleep(0.3); st.rerun()

        bb = mold_trimesh.bounds
        st.caption(
            f"📐 Bounds: X[{bb[0][0]:.1f},{bb[1][0]:.1f}] "
            f"Y[{bb[0][1]:.1f},{bb[1][1]:.1f}] "
            f"Z[{bb[0][2]:.1f},{bb[1][2]:.1f}] mm | "
            f"Voxel res: {res_mm_ui:.2f} mm | "
            f"Total voxels: {cache['total']:,}" if cache else ""
        )

elif os.path.exists(vtk_dir) and mold_trimesh is None:
    st.warning("⚠️ Upload STL file to enable 3D solid voxel animation.")
    st.info("📁 VTK results found — upload STL to visualize.")
else:
    st.info("📁 No VTK directory found. Run a simulation and click 'Sync Results'.")

st.divider()
st.caption("MIM-Ops Pro v2.4 | 3D Solid Voxel Fill | BFS Gate-Origin | Min-Dim/5 Mesh | AI-Powered Injection Molding")
