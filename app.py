"""
  MIM-Ops Pro v3.1
=================
v3.0 Architecture Change:
  [1] Heavy voxel/flow computation REMOVED from Streamlit
  [2] GitHub Actions dispatched for all solver tasks
  [3] Streamlit = UI only (trigger + result display)
  [4] solver.py runs on GitHub Actions (2-core, 7GB RAM)
  [5] Results pulled via GitHub Artifacts API
  [6] AI-powered gate suggestion via Gemini API (in-app)
"""

import streamlit as st
import os, time, uuid, requests, shutil
from datetime import datetime
import numpy as np
import zipfile
import io
import glob
import re
import trimesh
import plotly.graph_objects as go
import pyvista as pv
import base64

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops Pro v3.0: GitHub Actions Cloud Solver")

# ── Secrets ──
ZAPIER_URL    = st.secrets.get("ZAPIER_URL", "")
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", "")
REPO_OWNER    = st.secrets.get("REPO_OWNER", "workshopcompany")
REPO_NAME     = st.secrets.get("REPO_NAME", "OpenFOAM-Injection-Automation")
GEMINI_KEY    = st.secrets.get("GEMINI_API_KEY", "")  # ← Gemini API key

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
_init("mat_name", "17-4PH")
_init("last_vel_mms", 80.0); _init("last_etime", 1.0)
_init("gx_final", 0.0); _init("gy_final", 0.0); _init("gz_final", 0.0)
_init("animation_playing", False); _init("current_frame", 0)
_init("vtk_files", [])
_init("last_synced_signal_id", None)
_init("executed_params", None)
_init("num_frames", 15)
_init("gate_ai_suggested", False)
_init("gh_run_url", None)
_init("result_frames", [])   # PNG frames downloaded from GitHub
_init("machine_ton", 50)          # 사출기 톤수
_init("screw_dia_mm", 28.0)       # 스크류 직경 (mm)
_init("barrel_dia_mm", 28.0)      # 바렐 직경 (mm) — 보통 스크류와 동일

# ───────────────────── Logging ─────────────────────
def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["sim_logs"].append(f"[{ts}] {msg}")
    if len(st.session_state["sim_logs"]) > 100:
        st.session_state["sim_logs"] = st.session_state["sim_logs"][-100:]

def clear_old_results():
    for path in ["VTK", "results.txt", "logs.zip", "frames", "simulation-results", "temp_results"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    st.session_state["vtk_files"] = []
    st.session_state["result_frames"] = []
    add_log("Cleared old simulation results.")

# ───────────────────── VTK helpers (display only) ─────────────────────
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
                return min(float(np.sum(arr > 0.05) / max(len(arr), 1)), 1.0)
    except Exception as e:
        add_log(f"VTK read error: {e}")
    return None

# ── 사출기 톤수별 표준 스크류/바렐 스펙 DB ──────────────────────
# 출처: 일반적인 MIM/플라스틱 사출기 표준 스펙 (Arburg, Engel, 동신 등)
# 형식: ton → (screw_dia_mm, barrel_dia_mm, max_shot_cm3)
MACHINE_SPECS = {
     30: ( 22,  22,   30),
     50: ( 28,  28,   60),
     80: ( 32,  32,  100),
    100: ( 36,  36,  150),
    130: ( 40,  40,  210),
    150: ( 42,  42,  250),
    180: ( 45,  45,  320),
    200: ( 48,  48,  380),
}
MACHINE_TONS = sorted(MACHINE_SPECS.keys())

def get_machine_spec(ton: int) -> dict:
    """톤수에 가장 가까운 표준 스펙 반환"""
    closest = min(MACHINE_SPECS.keys(), key=lambda t: abs(t - ton))
    sd, bd, ms = MACHINE_SPECS[closest]
    return {"screw_dia_mm": float(sd), "barrel_dia_mm": float(bd), "max_shot_cm3": ms}

def calc_theoretical_fill_time(mesh_obj, gate_dia, vel_mms,
                                screw_dia_mm=28.0):
    """
    스크류 전진 속도(vel_mms) → 실제 게이트 통과 유량으로 보정한 충진시간 계산.

    원리:
      유량 보존 법칙: A_screw × v_screw = A_gate × v_gate
      → flow_rate(mm³/s) = A_screw × v_screw
      (게이트 단면적과 무관하게 스크류 기준 유량이 일정)
    """
    try:
        vol_mm3 = abs(mesh_obj.volume)
        if vol_mm3 <= 0 or gate_dia <= 0 or vel_mms <= 0 or screw_dia_mm <= 0:
            return 1.0
        screw_area = np.pi * ((screw_dia_mm / 2.0) ** 2)   # mm²
        flow_rate  = screw_area * vel_mms                   # mm³/s (실제 유량)
        return float(vol_mm3 / flow_rate)
    except Exception:
        return 1.0

# ───────────────────── Plotly traces ─────────────────────
def make_mold_trace(mold_trimesh, opacity=0.1):
    v, f = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2],
                     i=f[:,0], j=f[:,1], k=f[:,2],
                     opacity=opacity, color="lightgray",
                     name="Mold", showlegend=True)

# ───────────────────── Summary ─────────────────────
def build_summary_text():
    ep = st.session_state.get("executed_params")
    if ep is None:
        return None
    lines = [
        "Simulation Status: Dispatched to GitHub Actions",
        f"Material: {ep.get('material','N/A')}",
        f"Velocity: {ep.get('vel_mms',0)/1000:.4f} m/s   ({ep.get('vel_mms',0):.1f} mm/s)",
        f"Viscosity: {ep.get('viscosity',0):.2e} m²/s",
        f"Density: {ep.get('density',0):.0f} kg/m³",
        f"Melt Temp: {ep.get('melt_temp',0):.1f} °C",
        f"Injection Temp: {ep.get('temp',0):.1f} °C",
        f"Pressure: {ep.get('press',0):.1f} MPa",
        f"End Time: {ep.get('etime',0):.2f} s",
        f"Gate Dia: {ep.get('gate_dia',0):.1f} mm",
        f"Signal ID: {ep.get('signal_id','N/A')}",
    ]
    if st.session_state.get("gh_run_url"):
        lines.append(f"GitHub Run: {st.session_state['gh_run_url']}")
    if os.path.exists("results.txt"):
        with open("results.txt") as f:
            raw = f.read()
        for kw in ["Last Time Step", "Time Steps", "Finish Time"]:
            m = re.search(rf"{kw}[:\s]+(.+)", raw)
            if m:
                lines.append(f"{kw}: {m.group(1).strip()}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════
#  ★★★ MATERIAL DB ★★★
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=10)
def load_material_db(filepath: str) -> dict:
    db = {}
    if not os.path.exists(filepath):
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
                    continue
    except Exception as e:
        st.warning(f"material_property.txt load error: {e}")
    return db

def get_props(material: str) -> dict:
    name = material.upper().strip()
    db   = load_material_db(MATERIAL_FILE)
    if name in db:
        return {**db[name], "material": name, "source": "material_property.txt (exact)"}
    candidates = [k for k in db if name in k or k in name]
    if candidates:
        best = candidates[0]
        return {**db[best], "material": best, "source": f"material_property.txt (partial: '{best}')"}
    return {
        "nu": 1e-3, "rho": 1000.0, "Tmelt": 220.0, "Tmold": 50.0,
        "press_mpa": 70.0, "vel_mms": 80.0,
        "material": material, "source": "Default (not found in DB)"
    }

def get_process(material: str) -> dict:
    p = get_props(material)
    return {"temp": float(p["Tmelt"]), "press": float(p["press_mpa"]), "vel": float(p["vel_mms"])}

def list_known_materials() -> list:
    return sorted(load_material_db(MATERIAL_FILE).keys())

def save_material_to_txt(name: str, props: dict) -> bool:
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
        load_material_db.clear()
        return True
    except Exception as e:
        st.error(f"Material save error: {e}")
        return False

# ═══════════════════════════════════════════════════════════
#  ★★★ AI GATE SUGGESTION (Gemini API) ★★★
# ═══════════════════════════════════════════════════════════
def suggest_gate_positions_ai(mesh_obj: trimesh.Trimesh) -> list:
    """
    Geometric gate suggestion (local).
    If GEMINI_KEY is set, additionally queries Gemini 2.5 Flash Lite for material-aware advice.
    """
    suggestions = []
    try:
        bb     = mesh_obj.bounds
        dims   = bb[1] - bb[0]
        center = mesh_obj.centroid

        pt1 = np.array([center[0], center[1], bb[0][2]])
        snap1, _, _ = trimesh.proximity.closest_point(mesh_obj, [pt1])
        suggestions.append({"label": "Bottom-Center", "pos": snap1[0].tolist()})

        axis = int(np.argmax(dims))
        pt2  = center.copy(); pt2[axis] = bb[0][axis]
        snap2, _, _ = trimesh.proximity.closest_point(mesh_obj, [pt2])
        axis_label = ["X-Min Side", "Y-Min Side", "Z-Min Side"][axis]
        suggestions.append({"label": axis_label, "pos": snap2[0].tolist()})

        pt3 = np.array([center[0], center[1], bb[1][2]])
        snap3, _, _ = trimesh.proximity.closest_point(mesh_obj, [pt3])
        suggestions.append({"label": "Top-Center (Balanced)", "pos": snap3[0].tolist()})

    except Exception as e:
        add_log(f"Gate geometry analysis error: {e}")

    # ── Gemini API – advisory text only (no extra suggestion points) ──
    if GEMINI_KEY and st.session_state.get("props"):
        try:
            p = st.session_state["props"]
            prompt = (
                f"MIM injection molding. Material: {p.get('material','unknown')}, "
                f"viscosity={p.get('nu',0):.2e} m²/s, density={p.get('rho',0):.0f} kg/m³, "
                f"part volume≈{abs(mesh_obj.volume):.1f} mm³, "
                f"bounding box {np.array(mesh_obj.bounds[1])-np.array(mesh_obj.bounds[0])} mm. "
                f"Geometric gate candidates: {[s['label'] for s in suggestions]}. "
                f"In 2 sentences, recommend the best gate location and explain why for MIM."
            )

            # Gemini 2.5 Flash Lite API call
            gemini_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash-lite:generateContent?key={GEMINI_KEY}"
            )
            resp = requests.post(
                gemini_url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {
                        "maxOutputTokens": 200,
                        "temperature": 0.3,
                    }
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                advice = (
                    data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                        .strip()
                )
                if advice:
                    st.session_state["gate_ai_advice"] = advice
                    add_log("Gemini AI gate advisory received.")
            else:
                add_log(f"Gemini API error: HTTP {resp.status_code} — {resp.text[:200]}")
        except Exception as e:
            add_log(f"Gemini API exception: {e}")

    return suggestions

# ═══════════════════════════════════════════════════════════
# [1] 이 함수를 trigger_github_simulation 위에 추가하세요.
def upload_stl_to_github(file_bytes, target_path="input/part.stl"):
    """GitHub Contents API를 사용하여 파일을 레포지토리에 덮어씁니다."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{target_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 기존 파일 SHA 확인 (덮어쓰기를 위해 필수)
    resp = requests.get(url, headers=headers)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    # 업로드용 데이터 구성
    content_b64 = base64.b64encode(file_bytes).decode("utf-8")
    data = {
        "message": f"Upload STL for simulation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": content_b64,
        "branch": "main"  # 본인의 기본 브랜치가 main인지 확인하세요
    }
    if sha: data["sha"] = sha 

    put_resp = requests.put(url, headers=headers, json=data)
    return put_resp.status_code in [200, 201]
# ═══════════════════════════════════════════════════════════
#  ★★★ GITHUB ACTIONS TRIGGER ★★★
# ═══════════════════════════════════════════════════════════
def trigger_github_simulation(payload: dict) -> bool:
    """
    Triggers the GitHub Actions workflow via repository_dispatch.
    Returns True on success.
    """
    if not GITHUB_TOKEN:
        st.error("GITHUB_TOKEN not set in secrets.")
        return False
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
    # Strip internal-only keys (prefixed with _ or local display fields)
    # to stay within GitHub's 10-property client_payload limit.
    INTERNAL_KEYS = {"_gate_x", "_gate_y", "_gate_z", "_gate_dia",
                     "_num_frames", "_mesh_res_mm", "melt_temp", "gate_dia"}
    gh_payload = {k: v for k, v in payload.items() if k not in INTERNAL_KEYS}
    body = {
        "event_type": "run-simulation",
        "client_payload": gh_payload,
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=15)
        if r.status_code == 204:
            add_log(f"✅ GitHub Actions triggered | Signal: {payload['signal_id']}")
            return True
        else:
            st.error(f"GitHub dispatch failed: HTTP {r.status_code} — {r.text}")
            add_log(f"GitHub dispatch error: {r.status_code}")
            return False
    except Exception as e:
        st.error(f"GitHub dispatch exception: {e}")
        return False

def get_latest_run_url() -> str | None:
    """Returns the HTML URL of the most recent Actions run."""
    if not GITHUB_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs?per_page=1&event=repository_dispatch"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            if runs:
                return runs[0].get("html_url")
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════
#  ★★★ GITHUB ARTIFACT SYNC ★★★
# ═══════════════════════════════════════════════════════════
def sync_simulation_results():
    if not GITHUB_TOKEN:
        st.error("GITHUB_TOKEN not set in secrets.")
        return False
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    try:
        with st.spinner("Fetching results from GitHub Actions..."):
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                st.error(f"GitHub API error: {resp.status_code}")
                return False
            artifacts = resp.json().get("artifacts", [])

            # Prefer artifact matching current signal_id
            sig_id = st.session_state.get("last_signal_id", "")
            target = next(
                (a for a in artifacts if sig_id and sig_id in a.get("name", "")),
                next((a for a in artifacts if "simulation" in a.get("name", "").lower()), None)
            )
            if not target:
                st.warning("No simulation artifacts found yet. The job may still be running.")
                return False

            file_resp = requests.get(
                target["archive_download_url"], headers=headers, timeout=60
            )
            if file_resp.status_code == 200:
                clear_old_results()

                # ── 아티팩트를 'simulation-results/' 폴더로 추출 ──────────────
                # 진단 섹션(target_dirs)과 일치하는 경로를 사용해 bytes/path 불일치 해소
                extract_dir = "simulation-results"
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(file_resp.content)) as z:
                    z.extractall(extract_dir)

                # ── 루트 레벨 파일 편의 복사 (results.txt 등) ─────────────────
                for fname in ["results.txt", "results.json"]:
                    src = os.path.join(extract_dir, fname)
                    if os.path.exists(src):
                        shutil.copy(src, fname)

                # ── VTK 처리 ──────────────────────────────────────────────────
                vtk_dir = os.path.join(extract_dir, "VTK")
                if not os.path.exists(vtk_dir):
                    vtk_dir = "VTK"  # 루트에 있을 경우 폴백
                if os.path.exists(vtk_dir):
                    nf = st.session_state.get("num_frames", 15)
                    st.session_state["vtk_files"] = sample_vtk_files(vtk_dir, nf)

                # ── PNG 프레임 경로 수집 ────────────────────────────────────────
                # simulation-results/frames/ 우선, 없으면 simulation-results/ 전체 탐색
                pngs = sorted(
                    glob.glob(os.path.join(extract_dir, "frames", "frame_*.png")) or
                    glob.glob(os.path.join(extract_dir, "**", "frame_*.png"), recursive=True),
                    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
                    if re.findall(r'\d+', os.path.basename(x)) else 0
                )
                # 경로(str) 그대로 저장 — 진단 섹션, animation 섹션 모두 경로 기준으로 통일
                st.session_state["result_frames"] = pngs
                if pngs:
                    add_log(f"Loaded {len(pngs)} animation frames from GitHub.")
                else:
                    add_log("⚠️ No frame_*.png found in artifact.")

                # ── Signal ID 파싱 ─────────────────────────────────────────────
                if os.path.exists("results.txt"):
                    with open("results.txt") as f:
                        content = f.read()
                    m = re.search(r"Signal ID[:\s]+([A-Za-z0-9\-]+)", content)
                    if m:
                        st.session_state["last_synced_signal_id"] = m.group(1)

                st.session_state["sim_running"] = False
                st.session_state["sim_status"]  = "complete"
                add_log("✅ Results synchronized from GitHub Artifacts.")
                st.success(f"Results synced! Artifact: {target['name']} | Frames: {len(pngs)}")
                return True
            else:
                st.error(f"Failed to download artifact: HTTP {file_resp.status_code}")
                return False
    except Exception as e:
        st.error(f"Sync error: {e}")
        return False

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    if uploaded:
        try:
            # [추가] 새로운 파일이 올라오면 기존 컨펌 상태 리셋
            # 이렇게 해야 버튼이 안 먹히는 현상을 방지하고 새로 컨펌을 유도합니다.
            if "last_uploaded_name" not in st.session_state or st.session_state["last_uploaded_name"] != uploaded.name:
                st.session_state["props_confirmed"] = False
                st.session_state["process_confirmed"] = False
                st.session_state["last_uploaded_name"] = uploaded.name
                add_log(f"New file detected: {uploaded.name}. Resetting confirmations.")

            # 1. 3D 뷰어용 메쉬 로드
            uploaded.seek(0) 
            mesh_obj = trimesh.load(uploaded, file_type="stl")
            st.session_state["mesh"] = mesh_obj
            st.session_state["gate_ai_suggested"] = False

            # 2. GitHub 전송용 Base64 변환 (항상 하나의 데이터로 취급)
            uploaded.seek(0) 
            stl_bytes = uploaded.read()
            st.session_state["stl_b64"] = base64.b64encode(stl_bytes).decode('utf-8')
            
            st.success(f"✅ STL loaded — {len(mesh_obj.faces):,} faces")
        except Exception as e:
            st.error(f"STL load failed: {e}")

    st.divider()

    # ── Gate ──
    st.header("📍 2. Gate Configuration")
    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")
    mesh_obj = st.session_state.get("mesh")

    if mesh_obj:
        if st.button("🤖 AI Gate Suggest", use_container_width=True, type="secondary"):
            with st.spinner("Analyzing geometry..."):
                suggestions = suggest_gate_positions_ai(mesh_obj)
            if suggestions:
                st.session_state["gate_suggestions"] = suggestions
                best = suggestions[0]["pos"]
                st.session_state["gx"] = float(best[0])
                st.session_state["gy"] = float(best[1])
                st.session_state["gz"] = float(best[2])
                st.session_state["gate_ai_suggested"] = True
                add_log(f"AI Gate: {suggestions[0]['label']} → ({best[0]:.2f}, {best[1]:.2f}, {best[2]:.2f})")
                st.toast(f"✅ AI Suggestion: {suggestions[0]['label']}", icon="📍")
                st.rerun()

        if advice := st.session_state.get("gate_ai_advice"):
            st.info(f"🤖 AI Advisory: {advice}")

        suggestions = st.session_state.get("gate_suggestions", [])
        if suggestions:
            with st.expander("📌 Select AI Suggested Gate",
                             expanded=st.session_state.get("gate_ai_suggested", False)):
                for i, s in enumerate(suggestions):
                    p = s["pos"]
                    if st.button(f"{s['label']}  ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",
                                 key=f"gate_pick_{i}", use_container_width=True):
                        st.session_state["gx"] = float(p[0])
                        st.session_state["gy"] = float(p[1])
                        st.session_state["gz"] = float(p[2])
                        st.session_state["gate_ai_suggested"] = False
                        add_log(f"Gate selected: {s['label']}")
                        st.rerun()

    vx = st.number_input("Gate X", value=float(st.session_state["gx"]), step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=float(st.session_state["gy"]), step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=float(st.session_state["gz"]), step=0.1, key="gz")

    if mesh_obj:
        snap, _, _ = trimesh.proximity.closest_point(mesh_obj, [[vx, vy, vz]])
        gx, gy, gz = float(snap[0][0]), float(snap[0][1]), float(snap[0][2])
    else:
        gx, gy, gz = vx, vy, vz
    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz

    st.divider()

    # ── Material ──
    st.header("🧪 3. Material")
    known = list_known_materials()
    mat_name_input = st.text_input(
        "Material Name",
        value=st.session_state["mat_name"],
        key="mat_name_input",
        help=f"DB: {', '.join(known[:8])}{'...' if len(known) > 8 else ''}"
    )
    st.session_state["mat_name"] = mat_name_input

    col_ai, col_db = st.columns(2)
    with col_ai:
        if st.button("🚀 Run Simulation", key="run_sim_unique_key"):
            found = get_props(mat_name_input)
            st.session_state["props"] = found
            st.session_state["props_confirmed"] = False
            src = found.get("source", "")
            if "not found" in src:
                st.warning(f"⚠️ '{mat_name_input}' not found. Applying default values.")
                add_log(f"Material not found: {mat_name_input}")
            else:
                st.toast(f"✅ Material loaded: {src}", icon="🧪")
                add_log(f"Material loaded: {found['material']} ({src})")
    with col_db:
        if st.button("📋 DB List", use_container_width=True):
            st.session_state["show_material_list"] = not st.session_state.get("show_material_list", False)

    if st.session_state.get("show_material_list", False):
        with st.expander("📦 Materials in DB", expanded=True):
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
            st.caption(f"Source: {p.get('source','')}")
            p["nu"]    = st.number_input("Viscosity (m²/s)", value=float(p["nu"]),    format="%.2e")
            p["rho"]   = st.number_input("Density (kg/m³)",  value=float(p["rho"]))
            p["Tmelt"] = st.number_input("Melt Temp (°C)",   value=float(p["Tmelt"]))
            p["Tmold"] = st.number_input("Mold Temp (°C)",   value=float(p.get("Tmold", 50.0)))
            c_confirm, c_save = st.columns(2)
            with c_confirm:
                if st.button("✅ Confirm", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Properties confirmed!", icon="✅")
            with c_save:
                if st.button("💾 Save to DB", use_container_width=True):
                    mat_key = p.get("material", st.session_state["mat_name"]).upper().strip()
                    if save_material_to_txt(mat_key, {
                        "nu": p["nu"], "rho": p["rho"], "Tmelt": p["Tmelt"],
                        "Tmold": p.get("Tmold", 50.0),
                        "press_mpa": p.get("press_mpa", 100.0),
                        "vel_mms":   p.get("vel_mms", 50.0),
                    }):
                        st.toast(f"💾 '{mat_key}' saved!", icon="💾")

    st.divider()

    # ── Process ──
    st.header("⚙️ 4. Process")

    # ── 사출기 선택 ──
    st.markdown("**🏭 사출기 설정**")
    machine_ton = st.select_slider(
        "사출기 톤수 (ton)",
        options=MACHINE_TONS,
        value=st.session_state.get("machine_ton", 50),
        key="machine_ton"
    )
    # 톤수 변경 시 스펙 자동 업데이트
    spec = get_machine_spec(machine_ton)
    if st.session_state.get("_last_ton") != machine_ton:
        st.session_state["screw_dia_mm"]  = spec["screw_dia_mm"]
        st.session_state["barrel_dia_mm"] = spec["barrel_dia_mm"]
        st.session_state["_last_ton"]     = machine_ton

    mc1, mc2 = st.columns(2)
    with mc1:
        screw_dia = st.number_input(
            "스크류 직경 (mm)", 10.0, 150.0,
            value=float(st.session_state.get("screw_dia_mm", spec["screw_dia_mm"])),
            step=0.5, key="screw_dia_mm"
        )
    with mc2:
        barrel_dia = st.number_input(
            "바렐 직경 (mm)", 10.0, 150.0,
            value=float(st.session_state.get("barrel_dia_mm", spec["barrel_dia_mm"])),
            step=0.5, key="barrel_dia_mm"
        )
    st.caption(f"표준 스펙 ({machine_ton}ton): 스크류 ø{spec['screw_dia_mm']:.0f}mm | 최대 사출량 {spec['max_shot_cm3']}cm³")
    st.divider()

    mesh_obj = st.session_state.get("mesh")
    theo_time = 1.0
    if mesh_obj:
        vel_current = float(st.session_state["vel"])
        theo_time = calc_theoretical_fill_time(
            mesh_obj, float(g_size), vel_current,
            screw_dia_mm=float(st.session_state.get("screw_dia_mm", 28.0))
        )
        safe_etime_preview = min(theo_time * 1.5, 180.0)
        vol_mm3 = abs(mesh_obj.volume)
        screw_area = np.pi * ((float(st.session_state.get("screw_dia_mm", 28.0)) / 2.0) ** 2)
        flow_rate_cm3s = screw_area * vel_current / 1000.0
        st.info(
            f"💡 Est. Fill Time: **{theo_time:.2f}s**\n\n"
            f"→ Recommended End Time (×1.5): **{safe_etime_preview:.2f}s**\n\n"
            f"부품 체적: {vol_mm3:.0f} mm³ | 유량: {flow_rate_cm3s:.1f} cm³/s"
        )

    if st.button("🤖 Optimize Process", use_container_width=True):
        opt = get_process(mat_name_input)
        st.session_state.update({"temp": opt["temp"], "press": opt["press"], "vel": opt["vel"]})
        new_theo = calc_theoretical_fill_time(
            mesh_obj, float(g_size), opt["vel"],
            screw_dia_mm=float(st.session_state.get("screw_dia_mm", 28.0))
        ) if mesh_obj else 1.0
        safe_etime = min(new_theo * 1.5, 180.0)
        st.session_state["etime"] = safe_etime
        st.toast(f"Optimized! End Time: {safe_etime:.1f}s", icon="🤖")

    temp_c    = st.number_input("Temp (°C)",       50.0, 450.0, value=float(st.session_state["temp"]),  step=1.0, key="temp")
    press_mpa = st.number_input("Pressure (MPa)",  10.0, 250.0, value=float(st.session_state["press"]), step=1.0, key="press")
    vel_mms   = st.number_input("Velocity (mm/s)",  1.0, 600.0, value=float(st.session_state["vel"]),   step=1.0, key="vel")
    etime     = st.number_input(
        "End Time (s)", min_value=0.1, max_value=180.0,
        value=min(float(st.session_state["etime"]), 180.0),
        step=0.1, key="etime",
        help="Auto = 1.5× theoretical fill time. Max 180s."
    )

    if st.button("✅ Confirm Process", use_container_width=True):
        st.session_state["process_confirmed"] = True
        st.toast("Process confirmed!", icon="✅")

    st.divider()
    num_frames_sel = st.select_slider(
        "Animation Frames", options=[5, 10, 15, 20, 30],
        value=st.session_state.get("num_frames", 15)
    )
    st.session_state["num_frames"] = num_frames_sel

    st.divider()

    # ── Run button (사이드바 내부) ──
# [2] Run 버튼 로직을 아래 내용으로 교체하세요.
if st.sidebar.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True):
    if "stl_b64" not in st.session_state:
        st.error("⚠️ STL 파일을 먼저 업로드해주세요.")
    else:
        with st.spinner("1/2: GitHub에 STL 파일 업로드 중..."):
            stl_bytes = base64.b64decode(st.session_state["stl_b64"])
            # 'input/part.stl' 경로로 강제 고정 업로드
            upload_success = upload_stl_to_github(stl_bytes, "input/part.stl")

        if upload_success:
            st.success("✅ 1단계: STL 업로드 완료!")
            with st.spinner("2/2: 시뮬레이션 트리거 중..."):
                sig_id = str(uuid.uuid4())[:8]
                # 페이로드에는 텍스트 데이터만 담아서 64KB 제한 우회
                _gx    = float(st.session_state.get("gx_final", 0.0))
                _gy    = float(st.session_state.get("gy_final", 0.0))
                _gz    = float(st.session_state.get("gz_final", 0.0))
                _nf    = int(st.session_state.get("num_frames", 15))
                _screw = float(st.session_state.get("screw_dia_mm", 28.0))
                ep = {
                    "signal_id": sig_id,
                    "gate_pos":  f"{_gx:.4f},{_gy:.4f},{_gz:.4f},{float(g_size):.4f}",
                    "sim_opts":  f"{st.session_state['mat_name']},{_nf},0.5,{_screw}",
                    "viscosity": float(st.session_state["props"]["nu"]),
                    "density":   float(st.session_state["props"]["rho"]),
                    "temp":      float(temp_c),
                    "press":     float(press_mpa),
                    "vel_mms":   float(vel_mms),
                    "etime":     float(etime),
                }
                if trigger_github_simulation(ep):
                    st.toast("🚀 시뮬레이션 시작!", icon="✅")
                    time.sleep(2)
                    st.rerun()
        else:
            st.error("❌ 1단계: STL 파일 업로드에 실패했습니다. GitHub 토큰 권한을 확인하세요.")


# ═══════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════
mesh_obj = st.session_state.get("mesh")

col_geo, col_log = st.columns([2, 1])
with col_geo:
    st.header("🎥 3D Geometry & Gate")
    if mesh_obj:
        v, f = mesh_obj.vertices, mesh_obj.faces
        fig = go.Figure(data=[
            go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2],
                      i=f[:,0], j=f[:,1], k=f[:,2],
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
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode="data"),
            height=500,
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    else:
        st.info("Please upload an STL file.")

with col_log:
    st.header("📟 Simulation Logs")

    # Status badge
    status = st.session_state.get("sim_status", "idle")
    if status == "running":
        st.warning("⏳ **GitHub Actions running...** Click '🔄 Sync Results' when done.")
        if st.session_state.get("gh_run_url"):
            st.markdown(f"[🔗 Monitor on GitHub]({st.session_state['gh_run_url']})")
    elif status == "complete":
        st.success("✅ Simulation complete — results loaded.")
    else:
        st.info("Idle — configure & press Run.")

    lc = st.container(height=300)
    with lc:
        for log in st.session_state["sim_logs"][-25:]:
            st.code(log, language="bash")
    if st.button("🗑 Clear Logs", use_container_width=True):
        st.session_state["sim_logs"] = []
        st.rerun()

# ─────────── Results & Sync ───────────

# ─────────── [1] 초기화 (NameError 방지) ───────────
# 세션에서 가져오되, 없으면 빈 리스트를 기본값으로 설정하여 NameError 방지
if "result_frames" not in st.session_state:
    st.session_state["result_frames"] = []

# 로컬 변수로도 한 번 더 선언 (가장 확실한 방법)
current_frames = st.session_state["result_frames"]

st.title("📊 Simulation Results")

import os, glob, re, time, zipfile
# ─────────── [2] 시뮬레이션 결과 섹션 시작 ───────────
# ─────────── [2] 진단 로그 (압축 해제 상태 확인) ───────────
with st.expander("🔍 System Diagnostic Logs", expanded=True):
    # sync_simulation_results()가 추출하는 폴더와 동일한 경로 탐색
    target_dirs = ["simulation-results", "temp_results"]
    st.write(f"📂 **Checking Folders:** `{target_dirs}`")
    
    found_pngs = []
    for d in target_dirs:
        if os.path.exists(d):
            found_pngs.extend(glob.glob(os.path.join(d, "**", "*.png"), recursive=True))
    
    if found_pngs:
        found_pngs.sort(
            key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
            if re.findall(r'\d+', os.path.basename(x)) else 0
        )
        # 세션 상태와 동기화 (경로 리스트로 통일)
        st.session_state["result_frames"] = found_pngs
        current_frames = found_pngs
        st.success(f"✅ {len(found_pngs)} frames detected on disk.")
    else:
        st.warning("⚠️ No images found. Please click 'Sync Results'.")

# 3. 🔄 Sync 버튼 로직
# ─────────── 3. Sync & Animation 로직 ───────────
if st.button("🔄 Sync Results", key="sync_final_btn_v5", type="primary"):
    sync_simulation_results()
    st.session_state["current_frame"] = 0
    st.rerun()

# ─────────── PNG Frame Animation ───────────

# 4. 🌊 애니메이션 출력 섹션

if current_frames:
    st.subheader("🌊 Flow Animation")
    total = len(current_frames)

    curr_idx = st.session_state.get("current_frame", 0)
    if curr_idx >= total:
        curr_idx = 0

    # ── 컨트롤 버튼 ──
    btn_col1, btn_col2, btn_col3, btn_col_info = st.columns([1, 1, 1, 4])
    play_clicked  = btn_col1.button("▶ Play",  use_container_width=True, key="anim_play")
    pause_clicked = btn_col2.button("⏸ Pause", use_container_width=True, key="anim_pause")
    reset_clicked = btn_col3.button("⏹ Reset", use_container_width=True, key="anim_reset")

    if reset_clicked:
        st.session_state["current_frame"] = 0
        st.session_state["animation_playing"] = False
        curr_idx = 0
    if pause_clicked:
        st.session_state["animation_playing"] = False
    if play_clicked:
        st.session_state["animation_playing"] = True

    # ── Step Slider (수동 조작 시 재생 중단) ──
    new_idx = st.slider("Step Slider", 0, total - 1,
                        value=st.session_state.get("current_frame", 0),
                        key="frame_slider")
    if new_idx != st.session_state.get("current_frame", 0):
        st.session_state["animation_playing"] = False
        st.session_state["current_frame"] = new_idx
        curr_idx = new_idx

    # ── 프레임 정보 표시 ──
    fill_pct = (curr_idx + 1) / total * 100
    btn_col_info.markdown(
        f"<div style='padding:6px 0; color:#888; font-size:0.85rem;'>"
        f"Frame {curr_idx + 1} / {total} &nbsp;|&nbsp; Fill {fill_pct:.1f}%</div>",
        unsafe_allow_html=True
    )

    # ── 이미지 플레이스홀더 (Play 루프에서 이 자리를 교체) ──
    img_placeholder = st.empty()
    status_placeholder = st.empty()

    def _show_frame(idx):
        fd = current_frames[idx]
        cap = f"Step {idx+1}/{total}  |  Fill {(idx+1)/total*100:.1f}%"
        if isinstance(fd, str):
            if os.path.exists(fd):
                img_placeholder.image(fd, caption=cap, use_container_width=True)
            else:
                img_placeholder.warning(f"Frame file not found: {fd}")
        else:
            img_placeholder.image(fd, caption=cap, use_container_width=True)

    # ── 재생 중이면 루프, 아니면 현재 프레임만 표시 ──
    if st.session_state.get("animation_playing", False):
        status_placeholder.info("⏵ 재생 중... (Pause 또는 Reset으로 멈춤)")
        for i in range(curr_idx, total):
            if not st.session_state.get("animation_playing", False):
                break
            _show_frame(i)
            st.session_state["current_frame"] = i
            time.sleep(1.0)
        else:
            # 끝까지 재생 완료
            st.session_state["animation_playing"] = False
            st.session_state["current_frame"] = 0
        status_placeholder.empty()
        st.rerun()
    else:
        _show_frame(st.session_state.get("current_frame", 0))

else:
    st.info("💡 Sync 버튼을 눌러 결과 이미지를 로드하세요.")


# ─────────── Material DB Management ───────────
with st.expander("🗂️ Material DB Management (material_property.txt)", expanded=False):
    st.caption(f"File: `{MATERIAL_FILE}`")
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
        st.warning("DB is empty.")

    st.markdown("**Add / Update Material**")
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        new_mat   = st.text_input("Material Name", key="new_mat_name")
        new_nu    = st.number_input("Viscosity (m²/s)", value=4e-3, format="%.2e", key="new_nu")
    with nc2:
        new_rho   = st.number_input("Density (kg/m³)",  value=7800.0, key="new_rho")
        new_tmelt = st.number_input("Melt Temp (°C)",    value=185.0,  key="new_tmelt")
    with nc3:
        new_tmold = st.number_input("Mold Temp (°C)",    value=40.0,   key="new_tmold")
        new_press = st.number_input("Press (MPa)",        value=110.0,  key="new_press")
        new_vel   = st.number_input("Vel (mm/s)",         value=25.0,   key="new_vel")
    if st.button("💾 Add/Update DB", type="primary", use_container_width=True):
        if new_mat.strip():
            ok = save_material_to_txt(new_mat.strip(), {
                "nu": new_nu, "rho": new_rho, "Tmelt": new_tmelt,
                "Tmold": new_tmold, "press_mpa": new_press, "vel_mms": new_vel
            })
            if ok:
                st.success(f"✅ '{new_mat.upper()}' → Saved")
                st.rerun()
        else:
            st.warning("Please enter a material name.")
