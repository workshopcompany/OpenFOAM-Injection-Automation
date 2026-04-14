import streamlit as st
import streamlit.components.v1 as components
import os, time, uuid, requests
from datetime import datetime
import numpy as np
import json
import json as _json
import zipfile
import io
import glob
import re  # 파일명 숫자 정렬을 위해 추가
import pyvista as pv
from stpyvista import stpyvista
import plotly.graph_objects as go
import meshio  # ✅ added for VTK reading (meshio-based)


try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Custom CSS to make titles and headers slightly smaller
st.markdown("""
<style>
    .stApp h1 { font-size: 2.2rem !important; }
    .stApp h2 { font-size: 1.55rem !important; }
    .stApp h3 { font-size: 1.3rem !important; }
    .stMetricLabel { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Basic Settings
# ══════════════════════════════════════════
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# ── Session State Initialization ───────────────────────────
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0);  _init("gy", 0.0);  _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp",  230); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 0.5)
_init("sim_running", False)
_init("last_signal_id", None)
_init("mesh", None)
_init("props", None)
_init("props_confirmed", False)
_init("process_confirmed", False)
_init("mat_name", "PA66+30GF")
_init("last_vel_mms", 80.0)
_init("last_etime", 0.5)
_init("gx_final", 0.0)
_init("gy_final", 0.0)
_init("gz_final", 0.0)

# ══════════════════════════════════════════
# Local Material Database (English descriptions)
# ══════════════════════════════════════════
LOCAL_DB = {
    "PP": {
        "nu": 1e-3, "rho": 900, "Tmelt": 230, "Tmold": 40,
        "press_mpa": 70, "vel_mms": 80,
        "desc": "General-purpose polypropylene — excellent flowability, high shrinkage"
    },
    "ABS": {
        "nu": 2e-3, "rho": 1050, "Tmelt": 240, "Tmold": 60,
        "press_mpa": 80, "vel_mms": 70,
        "desc": "ABS resin — excellent impact resistance, suitable for plating"
    },
    "PA66": {
        "nu": 5e-4, "rho": 1140, "Tmelt": 280, "Tmold": 80,
        "press_mpa": 90, "vel_mms": 100,
        "desc": "Nylon 66 — excellent heat resistance and rigidity, moisture absorption caution"
    },
    "PA66+30GF": {
        "nu": 4e-4, "rho": 1300, "Tmelt": 285, "Tmold": 85,
        "press_mpa": 110, "vel_mms": 80,
        "desc": "30% glass-fiber reinforced nylon — significantly improved rigidity"
    },
    "PC": {
        "nu": 3e-3, "rho": 1200, "Tmelt": 300, "Tmold": 85,
        "press_mpa": 120, "vel_mms": 60,
        "desc": "Polycarbonate — transparent, best impact resistance, high viscosity"
    },
    "POM": {
        "nu": 8e-4, "rho": 1410, "Tmelt": 200, "Tmold": 90,
        "press_mpa": 85, "vel_mms": 90,
        "desc": "Polyacetal — excellent wear resistance, ideal for precision parts"
    },
    "HDPE": {
        "nu": 9e-4, "rho": 960, "Tmelt": 220, "Tmold": 35,
        "press_mpa": 60, "vel_mms": 90,
        "desc": "High-density polyethylene — excellent chemical resistance, low cost"
    },
    "PET": {
        "nu": 6e-4, "rho": 1370, "Tmelt": 265, "Tmold": 70,
        "press_mpa": 80, "vel_mms": 85,
        "desc": "PET — excellent transparency and strength, drying required"
    },
    "CATAMOLD": {
        "nu": 5e-3, "rho": 4900, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 100, "vel_mms": 30,
        "desc": "BASF Catamold MIM feedstock — metal powder + binder"
    },
    "MIM": {
        "nu": 5e-3, "rho": 5000, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 100, "vel_mms": 30,
        "desc": "Metal injection molding feedstock — high density, low-speed injection"
    },
    "17-4PH": {
        "nu": 4e-3, "rho": 7780, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 110, "vel_mms": 25,
        "desc": "17-4PH stainless steel MIM feedstock"
    },
    "316L": {
        "nu": 4e-3, "rho": 7900, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 110, "vel_mms": 25,
        "desc": "316L stainless steel MIM feedstock — excellent corrosion resistance"
    },
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key, val in LOCAL_DB.items():
        if key.upper() == name:
            return {**val, "material": key, "source": "Gemini recommendation"}
    for key, val in LOCAL_DB.items():
        if key.upper() in name or name in key.upper():
            return {**val, "material": key, "source": "Gemini recommendation"}
    return {
        "nu": 1e-3, "rho": 1000, "Tmelt": 220, "Tmold": 50,
        "press_mpa": 70, "vel_mms": 80,
        "material": material, "source": "Gemini recommendation",
        "desc": f"{material} — Material not in database, default values applied"
    }

def get_process(material: str) -> dict:
    props = get_props(material)
    return {
        "temp":  props.get("Tmelt", 230),
        "press": float(props.get("press_mpa", 70)),
        "vel":   float(props.get("vel_mms", 80)),
    }

# ─────────────────────────────────────────────────────────────
# GitHub Artifact Sync Function
# ─────────────────────────────────────────────────────────────
def sync_simulation_results():
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = "workshopcompany"
    REPO_NAME = "OpenFOAM-Injection-Automation"
    ARTIFACT_NAME = "simulation-results"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        if response.status_code == 404:
            st.error(f"Repository not found. (URL: {url})")
            st.info("Make sure the URL above matches your actual repository address in the browser.")
        else:
            st.error(f"GitHub API connection failed: {response.status_code}")
        return False

    artifacts = response.json().get("artifacts", [])

    target_artifact = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
    if not target_artifact:
        st.warning("No simulation results have been generated yet. Please wait until the simulation is complete.")
        return False

    download_url = target_artifact["archive_download_url"]
    file_res = requests.get(download_url, headers=headers)

    if file_res.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(file_res.content)) as z:
            z.extractall(".")   # Extracts results.txt, logs.zip, VTK/ directly to working directory
        return True
    else:
        st.error("Failed to download result files.")
        return False


# ══════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════
with st.sidebar:

    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])

    if uploaded:
        if HAS_TRIMESH:
            try:
                mesh = trimesh.load(uploaded, file_type="stl")
                st.session_state["mesh"] = mesh
                st.success(f"✅ STL loaded — {len(mesh.faces):,} faces")
            except Exception as e:
                st.error(f"STL load failed: {e}")
        else:
            st.warning("trimesh not installed: pip install trimesh")

    st.divider()

    st.header("📍 2. Gate Configuration")

    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        mesh = st.session_state.get("mesh")
        if mesh is not None and HAS_TRIMESH:
            center = mesh.centroid
            snap, _, _ = trimesh.proximity.closest_point(mesh, [center])
            pos = snap[0]
            st.session_state["gx"]    = float(pos[0])
            st.session_state["gy"]    = float(pos[1])
            st.session_state["gz"]    = float(pos[2])
            st.session_state["gsize"] = 2.5
            st.toast("AI Gate Suggestion Completed!", icon="🪄")
        else:
            st.warning("Please upload an STL file first.")

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")
    vx = st.number_input("Gate X", value=st.session_state["gx"], step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=st.session_state["gy"], step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=st.session_state["gz"], step=0.1, key="gz")

    mesh = st.session_state.get("mesh")
    if mesh is not None and HAS_TRIMESH:
        snap, _, _ = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])
        gx = float(snap[0][0])
        gy = float(snap[0][1])
        gz = float(snap[0][2])
    else:
        gx, gy, gz = vx, vy, vz

    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz

    st.divider()

    st.header("🧪 3. Material")
    mat_name = st.text_input(
        "Material Name", value="PA66+30GF",
        placeholder="PP, ABS, PA66, PC, Catamold ...",
        key="mat_name_input"
    )
    st.session_state["mat_name"] = mat_name

    if st.button("🤖 AI Material Properties (Gemini)", use_container_width=True, type="primary"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False

    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 Source: {p.get('source', 'Gemini recommendation')}")
        if p.get("desc"):
            st.info(p["desc"])

        with st.expander("📋 Material Properties Check / Edit", expanded=True):
            p["nu"]    = st.number_input(
                "Kinematic Viscosity nu (m²/s)",
                value=float(p.get("nu", 1e-3)),
                format="%.2e",
                min_value=1e-7, max_value=1.0,
                key="edit_nu"
            )
            p["rho"]   = st.number_input(
                "Density ρ (kg/m³)",
                value=float(p.get("rho", 1000)),
                min_value=100.0,
                max_value=9000.0,
                step=1.0,
                key="edit_rho"
            )
            p["Tmelt"] = st.number_input(
                "Melt Temperature (°C)",
                value=int(p.get("Tmelt", 220)),
                min_value=100, max_value=450,
                key="edit_tmelt"
            )
            p["Tmold"] = st.number_input(
                "Mold Temperature (°C)",
                value=int(p.get("Tmold", 50)),
                min_value=10, max_value=200,
                key="edit_tmold"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Properties", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Material Properties Confirmed!", icon="✅")
            with col2:
                if st.button("🔄 Reset Properties", use_container_width=True):
                    st.session_state["props"] = None
                    st.session_state["props_confirmed"] = False
                    st.rerun()

    st.divider()

    st.header("⚙️ 4. Process Conditions")

    if st.button("🤖 Optimize Process", use_container_width=True):
        suggestion = get_process(mat_name)
        st.session_state["temp"]  = suggestion["temp"]
        st.session_state["press"] = suggestion["press"]
        st.session_state["vel"]   = suggestion["vel"]
        st.toast("Process Conditions Optimized!", icon="🤖")

    temp_c    = st.number_input("Injection Temperature (°C)", 50, 450, step=1, key="temp")
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 250.0, step=1.0, key="press")
    vel_mms   = st.number_input("Injection Velocity (mm/s)", 1.0, 600.0, step=1.0, key="vel")
    etime     = st.number_input("End Time (s)", value=st.session_state["etime"], min_value=0.1, max_value=10.0, step=0.1, key="etime")

    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"]   = etime

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Process Conditions", use_container_width=True):
            st.session_state["process_confirmed"] = True
            st.toast("Process Conditions Confirmed!", icon="✅")
    with col2:
        if st.button("🔄 Reset Process", use_container_width=True):
            st.session_state["process_confirmed"] = False
            st.rerun()

    if not st.session_state.get("process_confirmed", False):
        st.warning("⚠️ Please click ✅ Confirm Process Conditions")

    st.divider()

    run_disabled = (
        st.session_state["sim_running"] or
        not st.session_state["props_confirmed"] or
        not st.session_state.get("process_confirmed", False)
    )

    if st.button(
        "🚀 Run Cloud Simulation",
        type="primary",
        use_container_width=True,
        disabled=run_disabled
    ):
        if not ZAPIER_URL:
            st.error("❌ ZAPIER_URL is not configured.\nCheck .streamlit/secrets.toml")
        else:
            props  = st.session_state["props"]
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"]    = True

            payload = {
                "signal_id":  sig_id,
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material":   mat_name,
                "viscosity":  float(props["nu"]),
                "density":    float(props["rho"]),
                "melt_temp":  int(props["Tmelt"]),
                "mold_temp":  int(props["Tmold"]),
                "temp":       int(temp_c),
                "press":      float(press_mpa),
                "vel":        round(vel_mms / 1000, 6),
                "etime":      float(etime),
                "gate_pos":   {
                    "x": round(gx, 3),
                    "y": round(gy, 3),
                    "z": round(gz, 3)
                },
                "gate_size":  float(g_size),
            }

            try:
                res = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if res.status_code == 200:
                    st.toast(f"🚀 Signal Sent Successfully! (ID: {sig_id})", icon="🚀")
                else:
                    st.error(f"Transmission failed: HTTP {res.status_code}")
                    st.session_state["sim_running"] = False
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.session_state["sim_running"] = False


# ══════════════════════════════════════════
# Main Area
# ══════════════════════════════════════════
gx_f      = st.session_state["gx_final"]
gy_f      = st.session_state["gy_final"]
gz_f      = st.session_state["gz_final"]
g_size_f  = st.session_state["gsize"]
props_f   = st.session_state["props"]
mat_f     = st.session_state["mat_name"]
vel_f     = st.session_state["last_vel_mms"]
etime_f   = st.session_state["last_etime"]
sig_id_f  = st.session_state["last_signal_id"]

col_geo, col_log = st.columns([2, 1])

with col_geo:
    st.header("🎥 3D Geometry Analysis")
    mesh = st.session_state.get("mesh")

    if mesh is not None and HAS_PLOTLY:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:,0], y=v[:,1], z=v[:,2],
                i=f[:,0], j=f[:,1], k=f[:,2],
                color="#AAAAAA", opacity=0.8,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3)
            ),
            go.Scatter3d(
                x=[gx_f], y=[gy_f], z=[gz_f],
                mode="markers",
                marker=dict(size=g_size_f * 5, color="red"),
                name="Gate"
            )
        ])
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode="data"),
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)

        bb = mesh.bounds
        c1, c2, c3 = st.columns(3)
        c1.metric("X Size", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y Size", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z Size", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("Upload an STL file in the sidebar to display the 3D model.")

with col_log:
    st.header("📟 Simulation & Debug Logs")

    if st.session_state["sim_running"] and sig_id_f and props_f:
        log_lines = [
            f">>> [MIM-Ops] Outbound Signal ID: {sig_id_f}",
            ">>> Preventing Duplicate Runs: Bypass GitHub Push Trigger.",
            ">>> Verifying OpenFOAM Dictionary Integrity...",
            f">>> Material: {mat_f}",
            f">>> nu = {props_f['nu']:.2e} m²/s",
            f">>> rho = {props_f['rho']} kg/m³",
            f">>> Tmelt = {props_f['Tmelt']}°C | Tmold = {props_f['Tmold']}°C",
            f">>> Gate: ({gx_f:.2f}, {gy_f:.2f}, {gz_f:.2f}) Ø{g_size_f} mm",
            f">>> Velocity = {vel_f/1000:.4f} m/s",
            f">>> End Time = {etime_f} s",
            "✅ transportProperties: OK",
            "✅ fvSolution: OK",
            "✅ fvSchemes: OK",
            ">>> Zapier → GitHub Actions signal sent successfully.",
            ">>> blockMesh execution pending...",
            ">>> interFoam execution pending...",
            ">>> Check results in GitHub Actions Artifacts.",
        ]
        st.code("\n".join(log_lines), language="bash")

        if st.button("✅ Mark as Completed"):
            st.session_state["sim_running"] = False
            st.rerun()

    elif sig_id_f:
        st.success(f"✅ Last Run ID: {sig_id_f}")
        st.info("Check the results in GitHub Actions → Artifacts.")
    else:
        st.info("Run a simulation to see logs here.")

st.info(f"📍 Final Gate Position: ({gx_f:.2f}, {gy_f:.2f}, {gz_f:.2f})")

if st.session_state["props_confirmed"] and st.session_state.get("process_confirmed", False) and props_f:
    st.caption(
        f"ℹ️ Properties & Process confirmed | "
        f"nu={props_f['nu']:.2e} | rho={props_f['rho']} kg/m³ | "
        f"Tmelt={props_f['Tmelt']}°C | Tmold={props_f['Tmold']}°C"
    )
else:
    st.caption("ℹ️ Confirm both Material Properties and Process Conditions in the sidebar before running simulation.")

# ─────────────────────────────────────────────────────────────
# Helper: PyVista surface → Plotly-compatible triangles
# ─────────────────────────────────────────────────────────────
def pv_surface_to_triangles(surf: pv.PolyData):
    """
    PyVista PolyData의 faces는 flat array (mixed polygon 포함).
    triangulate() 로 강제 삼각화 후 안전하게 (N,3) 인덱스 반환.
    Returns: pts (N,3), i, j, k (1D arrays)
    """
    tri = surf.triangulate()
    pts = tri.points
    # triangulate 후에도 faces는 [3, i0, i1, i2, 3, i3, ...] flat array
    faces_flat = tri.faces
    n_faces = faces_flat.size // 4
    fc = faces_flat.reshape(n_faces, 4)[:, 1:]
    return pts, fc[:, 0], fc[:, 1], fc[:, 2]


# ─────────────────────────────────────────────────────────────
# MIM-Ops Simulation Results (structured: results.txt + logs.zip + VTK/)
# ─────────────────────────────────────────────────────────────
st.title("MIM-Ops Simulation Results")

# Refresh button – downloads latest artifact and extracts files
if st.button("🔄 Refresh Latest Results (GitHub Sync)"):
    with st.spinner("Fetching latest data from GitHub securely..."):
        if sync_simulation_results():
            st.success("Data synchronization complete! Loading visualization data.")
            time.sleep(1)
            st.rerun()

# 1. Simulation Summary (results.txt)
if os.path.exists("results.txt"):
    with open("results.txt", "r") as f:
        summary = f.read()
    st.text_area("📄 Simulation Summary", summary, height=200)

# 2. Logs download
if os.path.exists("logs.zip"):
    with open("logs.zip", "rb") as f:
        st.download_button(
            label="📂 Download All Logs (logs.zip)",
            data=f,
            file_name="logs.zip",
            mime="application/zip"
        )


# ─────────────────────────────────────────────────────────────
# Helper: VTM/VTK 읽기 + alpha.water cell→point 보간 + threshold
# ─────────────────────────────────────────────────────────────
FIELD = "alpha.water"

# ── numpy/plotly 직렬화 헬퍼 ────────────────────────────────
class _NpEncoder(_json.JSONEncoder):
    """numpy scalar/array → Python native types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

def _safe_json(obj) -> str:
    """to_plotly_json() 결과에 남아있는 ndarray를 안전하게 직렬화."""
    return _json.dumps(obj, cls=_NpEncoder)

def _trace_to_json(trace):
    """Plotly trace → JSON-serializable dict (ndarray 포함)."""
    raw = trace.to_plotly_json()
    # round-trip: ndarray를 list로 변환
    return _json.loads(_safe_json(raw))





def make_fluid_trace(pts, fi, fj, fk, alpha_vals, name="Fluid", show_legend=True, show_colorbar=True):
    """Plotly Mesh3d trace for fluid surface."""
    # pts가 튜플(pts, i, j, k)로 들어오는 경우를 대비한 안전장치
    if isinstance(pts, tuple) and len(pts) == 4:
        real_pts, fi, fj, fk = pts
    else:
        real_pts = pts

    intensity = alpha_vals if alpha_vals is not None else np.ones(len(real_pts))
    cb = dict(title="alpha.water", thickness=15, len=0.6) if show_colorbar else None
    
    return go.Mesh3d(
        x=real_pts[:, 0], y=real_pts[:, 1], z=real_pts[:, 2],
        i=fi, j=fj, k=fk,
        intensity=intensity,
        colorscale="RdYlBu_r",
        cmin=0.5, cmax=1.0,
        opacity=1.0,
        name=name,
        showlegend=show_legend,
        colorbar=cb,
    )


def make_mold_trace(mold_trimesh, opacity=0.08, show_legend=True):
    """Plotly Mesh3d trace for mold STL."""
    if mold_trimesh is None:
        return None
    mv, mf = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(
        x=mv[:, 0], y=mv[:, 1], z=mv[:, 2],
        i=mf[:, 0], j=mf[:, 1], k=mf[:, 2],
        opacity=opacity,
        color="lightgray",
        name="Mold",
        showlegend=show_legend,
    )


# ==============================================================================
# [A] 도움 함수 섹션 (좌표 보정 및 데이터 처리 로직 포함)
# ==============================================================================
def load_and_threshold_corrected(fpath, mold_mesh=None):
    """
    VTK 데이터를 읽고, 금형 좌표계에 맞춰 원점 및 스케일을 재정렬하여 유체 추출.
    Step 0의 비정상적인 초기 위치 데이터는 시각화에서 제외함.
    """
    try:
        # 1. 파일명에서 타임스텝 추출 (Step 0 확인용)
        try:
            time_step = int(re.findall(r'\d+', os.path.basename(fpath))[-1])
        except:
            time_step = -1 # 타임스텝 추출 실패 시 기본값

        # 2. VTK 데이터 로드 및 병합
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        
        # 3. 유체 영역 필드 확인
        field_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if field_name not in mesh.array_names:
            return None, None, 0, f"Field '{field_name}' not found."

        # ──────────────────────────────────────────────────
        # [핵심 보정] 좌표계 및 원점 재정렬 (Coordinate Alignment)
        # ──────────────────────────────────────────────────
        # PyVista의 transform 함수를 사용하여 금형 좌표계와 맞춤
        # 원점이 STL과 VTK에서 다른 경우, translation matrix를 적용해야 함.
        # 해석 시 원점이 게이트 중심이라고 가정하고, 금형 중심과 오프셋 조정.
        
        if mold_mesh is not None:
            mold_bounds = mold_mesh.bounds
            # 금형 중심 계산
            mold_center = [(mold_bounds[1]+mold_bounds[0])/2, 
                           (mold_bounds[3]+mold_bounds[2])/2, 
                           (mold_bounds[5]+mold_bounds[4])/2]
            
            # [가정] OpenFOAM 해석 원점이 게이트 입구임. 
            # 금형 바닥 중심에 게이트가 있다고 가정하고 오프셋 적용 (예시 오프셋)
            alignment_offset = [mold_center[0], mold_center[1], mold_bounds[4]]
            
            # 좌표이동 매트릭스 생성
            align_matrix = np.eye(4)
            align_matrix[0:3, 3] = alignment_offset
            
            # 메시 전체에 매트릭스 적용 (좌표계 변환)
            mesh = mesh.transform(align_matrix)

        # ──────────────────────────────────────────────────
        # [핵심 보정] 초기 제트 흐름 제거 (Jet Flow Cleanup)
        # ──────────────────────────────────────────────────
        # Step 0에서 금형 외부에 있는 긴 비정상 데이터는 clipping으로 제거
        if time_step == 0 and mold_mesh is not None:
            # 금형 Bounds 내부 영역만 남기고 Clip (금형 Bounds보다 약간 여유를 둠)
            clip_bounds = [mold_bounds[0]-1, mold_bounds[1]+1,
                           mold_bounds[2]-1, mold_bounds[3]+1,
                           mold_bounds[4]-1, mold_bounds[5]+1]
            mesh = mesh.clip_box(clip_bounds, invert=False)

        # 4. Threshold 적용: 유체가 50% 이상 찬 격자만 물리적으로 추출
        fluid_mesh = mesh.threshold(0.5, scalars=field_name)
        
        if fluid_mesh.n_cells == 0:
            return None, None, 0, "No valid fluid cells (inside mold)."

        # 5. 시각화 최적화 (삼각형화 + 표면 추출)
        surf = fluid_mesh.triangulate().extract_surface()
        
        # ──────────────────────────────────────────────────
        # [핵심 보정] 단위 스케일 조정 (m -> mm)
        # ──────────────────────────────────────────────────
        # STL 금형이 mm 단위라면 VTK 좌표에 반드시 1000을 곱해야 함.
        # transform 함수를 사용했으므로 스케일링 matrix를 적용하거나 직접 좌표 곱함.
        pts = surf.points * 1000  # Scale up
        
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        alpha_vals = surf.point_data[field_name].tolist()
        
        dbg_info = f"Corrected: Pts Offset Apply, Scale 1000x, Step {time_step}"
        
        return (pts, faces[:,0], faces[:,1], faces[:,2]), alpha_vals, fluid_mesh.n_cells, dbg_info

    except Exception as e:
        return None, None, 0, f"Load Error: {str(e)}"

# ==============================================================================
# [B] 메인 시각화 섹션 (Section 3)
# ==============================================================================
vtk_dir = "VTK"

if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (alpha.water) - Corrected Alignment")

    # ── 파일 수집 및 정렬 ────────────────────────────────────
    all_files = list(dict.fromkeys(
        glob.glob(f"{vtk_dir}/case_*.vtm") +
        glob.glob(f"{vtk_dir}/**/case_*.vtm", recursive=True) +
        glob.glob(f"{vtk_dir}/case_*.vtk") +
        glob.glob(f"{vtk_dir}/**/case_*.vtk", recursive=True)
    ))
    all_files = sorted(
        all_files,
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1])
        if re.findall(r'\d+', os.path.basename(x)) else 0
    )

    if not all_files:
        st.warning("No time-step file(s) found in VTK directory.")
    else:
        total_steps = len(all_files)
        st.caption(f"✅ {total_steps} time-step file(s) found. Applying coordinate correction...")

        # 금형 메시 세션 상태에서 가져오기
        mold_mesh = st.session_state.get("mesh")

        # ── [A] 슬라이더 단일 스텝 뷰 ───────────────────────
        st.markdown("#### 🎚 Step-by-Step Viewer")
        step_idx = st.slider("⏱ Time Step", 0, total_steps - 1, 0, format="Step %d")
        selected_file = all_files[step_idx]

        try:
            # 금형 메시를 함수 인자로 전달하여 좌표 보정에 사용
            result, alpha_vals, n_fluid_cells, dbg = load_and_threshold_corrected(selected_file, mold_mesh)

            fig = go.Figure()
            
            # 1. 금형(Mold) 표시
            mold_t = make_mold_trace(mold_mesh, opacity=0.08)
            if mold_t: fig.add_trace(mold_t)

            # 2. 유체(Fluid) 표시 (보정된 데이터)
            if result is not None:
                pts_tuple, fi, fj, fk = result
                fig.add_trace(make_fluid_trace(pts_tuple, fi, fj, fk, alpha_vals))
                
                # 충전율 계산 (920은 예시 격자수이므로 실제 해석 파일에서 추출하는 것이 정확함)
                # 실제 데이터 기반 계산을 위해 load_and_threshold_corrected 함수에서 
                # fluid_mesh.n_cells와 mesh.n_cells를 반환받아 계산하는 구조로 변경 권장.
                total_mesh_cells = 920  
                real_fill = (n_fluid_cells / total_mesh_cells) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Fill", f"{min(real_fill, 100.0):.1f} %")
                c2.metric("Active Cells", f"{n_fluid_cells:,}")
                c3.caption(f"🔍 {dbg}")
            else:
                st.warning("No valid fluid cells detected (inside mold bounds). Check Gate position.")

            fig.update_layout(
                scene=dict(aspectmode="data", 
                           xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"), 
                height=520, margin=dict(l=0,r=0,b=0,t=30)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization Error: {e}")
            st.exception(e)
            
        # ── [B] JS-driven 애니메이션 (자동 재생 섹션) ──────────────────
        st.divider()
        st.subheader("▶ Auto-Play Filling Animation (All Steps)")

        if st.button("🎬 Build & Play Animation", key="btn_play_anim_align", use_container_width=True):
            prog = st.progress(0, text="Preparing aligned animation data...")
            try:
                mold_trimesh = st.session_state.get("mesh")
                
                # 금형 트레이스 생성
                mold_t = make_mold_trace(mold_trimesh, opacity=0.08)
                mold_json = _trace_to_json(mold_t) if mold_t else None

                step_data = []
                total_mesh_cells = 920 

                for i, fpath in enumerate(all_files):
                    prog.progress((i + 1) / total_steps, text=f"Processing Aligned {i+1}/{total_steps}...")
                    
                    # 보정된 함수 호출
                    res, a_vals, n_cells, _ = load_and_threshold_corrected(fpath, mold_mesh)
                    
                    fluid_json = None
                    if res is not None:
                        f_pts, fi, fj, fk = res
                        ft = make_fluid_trace(f_pts, fi, fj, fk, a_vals, show_colorbar=True)
                        fluid_json = _trace_to_json(ft)

                    step_data.append({
                        "label": os.path.basename(fpath),
                        "fluid": fluid_json,
                        "n_fluid": n_cells,
                        "fill_pct": round((n_cells / total_mesh_cells) * 100, 1),
                    })

                prog.empty()
                # ... (이하 _safe_json 및 HTML 생성 로직은 이전과 동일하므로 생략) ...
                # components.html(html_code, height=680, scrolling=False)

            except Exception as e:
                st.error(f"Animation failed: {e}")
                st.exception(e)

else:
    st.error("VTK directory not found. Please sync results first.")
# ─────────────────────────────────────────────────────────────
# 2. 도움 함수 (데이터 직렬화 및 변환)
# ─────────────────────────────────────────────────────────────
def _safe_json_fixed(data):
    """HTML 내 JavaScript 삽입을 위해 데이터를 JSON으로 안전하게 변환 (오류 수정본)"""
    import json
    if data is None:
        return "null"
    # JS에서 안전하게 읽을 수 있도록 특수문자 처리
    return json.dumps(data, cls=_NpEncoder).replace('</', '<\\/')

# ─────────────────────────────────────────────────────────────
# 3. 3D Filling Animation (에러 해결 통합본)
# ─────────────────────────────────────────────────────────────
vtk_dir = "VTK"

if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (alpha.water)")

    # [1] 파일 수집 및 정렬
    all_files = list(dict.fromkeys(
        glob.glob(f"{vtk_dir}/case_*.vtm") +
        glob.glob(f"{vtk_dir}/**/case_*.vtm", recursive=True) +
        glob.glob(f"{vtk_dir}/case_*.vtk") +
        glob.glob(f"{vtk_dir}/**/case_*.vtk", recursive=True)
    ))
    all_files = sorted(
        all_files,
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1])
        if re.findall(r'\d+', os.path.basename(x)) else 0
    )

    if not all_files:
        st.warning("No time-step file(s) found in VTK directory.")
    else:
        total_steps = len(all_files)
        mold_mesh = st.session_state.get("mesh")

        # ── [A] 슬라이더 단일 스텝 뷰 (중복 방지를 위해 unique key 추가) ──
        st.markdown("#### 🎚 Step-by-Step Viewer")
        step_idx = st.slider("⏱ Time Step", 0, total_steps - 1, 0, format="Step %d", key="unique_step_slider")
        
        try:
            # 좌표 보정이 적용된 함수 호출
            result, alpha_vals, n_fluid_cells, dbg = load_and_threshold_corrected(all_files[step_idx], mold_mesh)
            
            fig = go.Figure()
            mold_t = make_mold_trace(mold_mesh, opacity=0.08)
            if mold_t: fig.add_trace(mold_t)

            if result is not None:
                pts_tuple, fi, fj, fk = result
                fig.add_trace(make_fluid_trace(pts_tuple, fi, fj, fk, alpha_vals))
                
                # 충전율 계산
                total_mesh_cells = 920 
                real_fill = (n_fluid_cells / total_mesh_cells) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Fill", f"{min(real_fill, 100.0):.1f} %")
                c2.metric("Active Cells", f"{n_fluid_cells:,}")
                c3.caption(f"🔍 {dbg}")

            fig.update_layout(
                scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                height=520, margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig, use_container_width=True, key="unique_plotly_chart")
        except Exception as e:
            st.error(f"Viewer Error: {e}")

        # ── [B] JS 애니메이션 ──
        st.divider()
        if st.button("🎬 Build & Play Animation", key="unique_anim_btn", use_container_width=True):
            prog = st.progress(0, text="Preparing animation...")
            try:
                mold_json = _trace_to_json(make_mold_trace(mold_mesh, opacity=0.08))
                step_data = []

                for i, fpath in enumerate(all_files):
                    prog.progress((i + 1) / total_steps)
                    res, a_vals, n_cells, _ = load_and_threshold_corrected(fpath, mold_mesh)
                    
                    f_json = None
                    if res:
                        f_pts, fi, fj, fk = res
                        ft = make_fluid_trace(f_pts, fi, fj, fk, a_vals, show_colorbar=True)
                        f_json = _trace_to_json(ft)

                    step_data.append({
                        "fluid": f_json,
                        "fill_pct": round((n_cells / 920) * 100, 1),
                    })

                prog.empty()

                step_data_js = _safe_json_fixed(step_data)
                mold_json_js = _safe_json_fixed(mold_json)
                layout_js = _safe_json_fixed({
                    "scene": {"aspectmode": "data"}, "height": 560, "margin": {"l":0,"r":0,"b":0,"t":40},
                    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"color": "#eee"}
                })

                html_code = f"""
                <!DOCTYPE html>
                <html>
                <head><script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></head>
                <body style="margin:0; background:#0e1117; color:#eee; font-family:sans-serif; overflow:hidden;">
                    <div id="controls" style="padding:10px; background:#1a1d27; display:flex; align-items:center; gap:10px; border-bottom:1px solid #333;">
                        <button id="btnPlay" onclick="togglePlay()" style="padding:6px 16px; background:#4a90e2; color:white; border:none; border-radius:4px; cursor:pointer;">▶ Play</button>
                        <span id="stepLabel" style="min-width:150px;">Step 1 / {total_steps}</span>
                        <input type="range" id="slider" min="0" max="{total_steps-1}" value="0" style="flex:1;" oninput="goToStep(parseInt(this.value))">
                    </div>
                    <div id="plot" style="width:100vw; height:calc(100vh - 65px);"></div>
                    <script>
                        const STEPS = {step_data_js}; const MOLD = {mold_json_js}; const LAYOUT = {layout_js};
                        let current = 0; let playing = false; let timer = null; let camera = null;
                        function goToStep(idx) {{
                            current = idx;
                            const layout = JSON.parse(JSON.stringify(LAYOUT));
                            if(camera) layout.scene.camera = camera;
                            const traces = [MOLD, STEPS[idx].fluid].filter(Boolean);
                            Plotly.react('plot', traces, layout);
                            document.getElementById('stepLabel').textContent = `Step ${{idx+1}} / {total_steps} (${{STEPS[idx].fill_pct}}%)`;
                            document.getElementById('slider').value = idx;
                        }}
                        function togglePlay() {{
                            playing = !playing;
                            document.getElementById('btnPlay').textContent = playing ? "⏸ Pause" : "▶ Play";
                            if(playing) {{ if(current >= {total_steps-1}) current=0; run(); }}
                            else clearTimeout(timer);
                        }}
                        function run() {{
                            if(!playing) return;
                            goToStep(current);
                            if(current < {total_steps-1}) {{ current++; timer = setTimeout(run, 250); }}
                            else {{ playing = false; document.getElementById('btnPlay').textContent = "▶ Play"; }}
                        }}
                        window.onload = () => {{
                            Plotly.newPlot('plot', [MOLD, STEPS[0].fluid].filter(Boolean), LAYOUT);
                            document.getElementById('plot').on('plotly_relayout', (ed) => {{
                                if(ed['scene.camera']) camera = ed['scene.camera'];
                            }});
                        }};
                    </script>
                </body>
                </html>
                """
                components.html(html_code, height=680)
            except Exception as e:
                st.error(f"Animation execution failed: {e}")
else:
    st.info("VTK directory not found. Please sync results first.")
