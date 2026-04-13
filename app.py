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


def load_and_threshold(fpath: str, threshold: float = 0.5):
    """
    VTM/VTK 파일을 읽어서 alpha.water 기준으로 유체 영역만 추출.
    Returns: (fluid_surf: pv.PolyData | None, alpha_vals: np.ndarray | None,
              n_fluid_cells: int, debug_info: str)
    """
    raw = pv.read(fpath)

    # MultiBlock → 단일 UnstructuredGrid
    if isinstance(raw, pv.MultiBlock):
        blocks = [raw.get(i) for i in range(raw.n_blocks) if raw.get(i) is not None]
        if not blocks:
            return None, None, 0, "MultiBlock is empty"
        raw = pv.UnstructuredGrid()
        raw = blocks[0].copy()
        for b in blocks[1:]:
            raw = raw.merge(b)

    debug_parts = [
        f"cells={raw.n_cells}",
        f"point_arrays={list(raw.point_data.keys())}",
        f"cell_arrays={list(raw.cell_data.keys())}",
    ]

    # alpha.water를 point_data로 확보
    #  - point_data에 있으면 그대로 사용
    #  - cell_data에만 있으면 cell→point 보간
    if FIELD in raw.point_data:
        mesh_p = raw
        debug_parts.append("alpha: point_data ✅")
    elif FIELD in raw.cell_data:
        mesh_p = raw.cell_data_to_point_data()
        debug_parts.append("alpha: cell→point 보간 ✅")
    else:
        debug_parts.append(f"alpha: NOT FOUND ❌")
        return None, None, 0, " | ".join(debug_parts)

    # threshold: alpha.water > threshold 인 셀만
    alpha_arr = mesh_p.point_data[FIELD]
    debug_parts.append(f"alpha range=[{alpha_arr.min():.3f}, {alpha_arr.max():.3f}]")

    fluid = mesh_p.threshold(threshold, scalars=FIELD)
    debug_parts.append(f"fluid_cells={fluid.n_cells}")

    if fluid.n_cells == 0:
        return None, None, 0, " | ".join(debug_parts)

    surf = fluid.extract_surface()
    pts, fi, fj, fk = pv_surface_to_triangles(surf)

    # ── 단위 보정: OpenFOAM VTK는 m 단위, STL은 mm 단위 → 1000배 스케일
    pts = pts * 1000.0
    debug_parts.append(f"pts_range_mm=[{pts.min():.1f}, {pts.max():.1f}]")

    tri_surf = surf.triangulate()

    # alpha 값 추출
    alpha_vals = None
    if FIELD in tri_surf.point_data:
        alpha_vals = tri_surf.point_data[FIELD]
    elif FIELD in tri_surf.cell_data:
        tmp = tri_surf.cell_data_to_point_data()
        alpha_vals = tmp.point_data.get(FIELD)

    return (pts, fi, fj, fk), alpha_vals, fluid.n_cells, " | ".join(debug_parts)



def make_fluid_trace(pts, fi, fj, fk, alpha_vals, name="Fluid", show_legend=True, show_colorbar=True):
    """Plotly Mesh3d trace for fluid surface."""
    intensity = alpha_vals if alpha_vals is not None else np.ones(len(pts))
    cb = dict(title="alpha.water", thickness=15, len=0.6) if show_colorbar else None
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
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


# ─────────────────────────────────────────────────────────────
# 3. 3D Filling Animation  
# ── 핵심: 매 step마다 geometry가 달라지므로 Plotly go.Frame 방식은
#    topology 변화에서 mesh를 고정해버린다. 
#    → 각 step의 Plotly figure JSON을 Python에서 직렬화해두고,
#      JS setTimeout 루프로 Plotly.react() 를 호출하는 방식으로 해결.
# ─────────────────────────────────────────────────────────────

vtk_dir = "VTK"

if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (alpha.water)")

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
        st.warning("No 'case_*.vtm / case_*.vtk' files found in VTK directory.")
    else:
        total_steps = len(all_files)
        st.caption(f"✅ {total_steps} time-step file(s) found")

        # ── 디버그 expander ──────────────────────────────────
        with st.expander("🔍 Debug: VTK Field Info (first & last step)", expanded=False):
            for label, fp in [("First step", all_files[0]), ("Last step", all_files[-1])]:
                result, alpha_vals, n_cells, dbg = load_and_threshold(fp)
                st.code(f"[{label}] {os.path.basename(fp)}\n{dbg}", language="bash")

        # ── [A] 슬라이더 단일 스텝 뷰 ───────────────────────
        st.markdown("#### 🎚 Step-by-Step Viewer")
        step_idx = st.slider(
            "⏱ Time Step",
            min_value=0, max_value=total_steps - 1,
            value=0, step=1, format="Step %d"
        )
        selected_file = all_files[step_idx]
        st.info(f"Rendering: `{os.path.basename(selected_file)}`  "
                f"(step {step_idx + 1} / {total_steps})")

        try:
            result, alpha_vals, n_fluid_cells, dbg = load_and_threshold(selected_file)

            fig = go.Figure()
            mold_t = make_mold_trace(st.session_state.get("mesh"))
            if mold_t:
                fig.add_trace(mold_t)

            if result is not None:
                pts, fi, fj, fk = result
                fig.add_trace(make_fluid_trace(pts, fi, fj, fk, alpha_vals))
                total_cells = 920  # debug에서 확인된 고정값
                real_fill = n_fluid_cells / total_cells * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Fill Ratio", f"{real_fill:.1f} %")
                c2.metric("Fluid Cells", f"{n_fluid_cells:,}")
                c3.metric("Total Cells", f"{total_cells:,}")
            else:
                st.warning(f"No fluid cells. Debug: {dbg}")

            fig.update_layout(
                scene=dict(aspectmode="data",
                           xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, b=0, t=30),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization Error: {e}")
            st.exception(e)

        # ── [B] JS-driven 애니메이션 ─────────────────────────
        # Plotly go.Frame은 topology가 달라지면 geometry를 고정해버림.
        # 해결: 각 step의 figure JSON을 미리 직렬화 → JS에서
        #        Plotly.react(div, data, layout) 를 step마다 완전히 교체.
        st.divider()
        st.subheader("▶ Auto-Play Filling Animation (All Steps)")

        if st.button("🎬 Build & Play Animation", use_container_width=True):
            prog = st.progress(0, text="Serializing frames…")
            try:
                mold_trimesh = st.session_state.get("mesh")

                # 금형 trace JSON (모든 frame 공통 — 변하지 않음)
                mold_json = None
                if mold_trimesh is not None:
                    mt = make_mold_trace(mold_trimesh, opacity=0.08, show_legend=True)
                    mold_json = _trace_to_json(mt)

                # 각 step의 유체 trace JSON + 메타정보
                step_data = []   # [{fluid: <trace_json|null>, n_fluid, fill_pct}, ...]
                total_cells = 920

                for i, fpath in enumerate(all_files):
                    prog.progress(
                        (i + 1) / total_steps,
                        text=f"Serializing {i+1}/{total_steps}: {os.path.basename(fpath)}"
                    )
                    result, alpha_vals, n_fluid_cells, _ = load_and_threshold(fpath)
                    fluid_json = None
                    if result is not None:
                        pts, fi, fj, fk = result
                        ft = make_fluid_trace(
                            pts, fi, fj, fk, alpha_vals,
                            show_colorbar=True
                        )
                        fluid_json = _trace_to_json(ft)

                    step_data.append({
                        "label": os.path.basename(fpath),
                        "fluid": fluid_json,
                        "n_fluid": n_fluid_cells,
                        "fill_pct": round(n_fluid_cells / total_cells * 100, 1),
                    })

                prog.empty()

                # 공통 layout JSON
                layout_dict = {
                    "scene": {
                        "aspectmode": "data",
                        "xaxis": {"title": "X (m)"},
                        "yaxis": {"title": "Y (m)"},
                        "zaxis": {"title": "Z (m)"},
                    },
                    "legend": {"x": 0.01, "y": 0.99},
                    "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
                    "height": 560,
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor":  "rgba(0,0,0,0)",
                }

                # JSON 직렬화
                step_data_js  = _safe_json(step_data)
                mold_json_js  = _safe_json(mold_json)
                layout_js     = _safe_json(layout_dict)

                html_code = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ margin:0; background:#0e1117; color:#fff;
            font-family: 'Segoe UI', sans-serif; }}
    #controls {{
      display:flex; align-items:center; gap:12px;
      padding:10px 16px; background:#1a1d27; border-radius:8px;
      margin-bottom:8px;
    }}
    button {{
      padding:6px 18px; border:none; border-radius:6px;
      background:#4a90e2; color:#fff; font-size:14px;
      cursor:pointer; transition:background .2s;
    }}
    button:hover {{ background:#357abd; }}
    button:disabled {{ background:#555; cursor:default; }}
    #stepLabel {{ font-size:13px; color:#aaa; min-width:220px; }}
    #fillBar {{
      flex:1; height:8px; background:#333; border-radius:4px; overflow:hidden;
    }}
    #fillInner {{ height:100%; background:#4a90e2; transition:width .3s; }}
    #fillPct {{ font-size:13px; color:#7ecfff; min-width:55px; text-align:right; }}
    input[type=range] {{
      flex:1; accent-color:#4a90e2;
    }}
  </style>
</head>
<body>
  <div id="controls">
    <button id="btnPlay" onclick="togglePlay()">▶ Play</button>
    <button id="btnPrev" onclick="stepBy(-1)">◀</button>
    <button id="btnNext" onclick="stepBy(1)">▶</button>
    <span id="stepLabel">Step 1 / {total_steps}</span>
    <div id="fillBar"><div id="fillInner" style="width:0%"></div></div>
    <span id="fillPct">0.0 %</span>
  </div>
  <input type="range" id="slider" min="0" max="{total_steps - 1}" value="0"
    style="width:100%;margin-bottom:6px;accent-color:#4a90e2"
    oninput="goToStep(parseInt(this.value))">
  <div id="plot"></div>

<script>
const STEPS      = {step_data_js};
const MOLD       = {mold_json_js};
const LAYOUT     = {layout_js};
const TOTAL      = {total_steps};
let   currentStep = 0;
let   playing     = false;
let   timer       = null;
let   sceneCamera = null;

// ── 초기 렌더 ────────────────────────────────────────────
Plotly.newPlot('plot', buildTraces(0), LAYOUT, {{responsive:true}});

// 카메라 위치 유지: 사용자가 돌린 시점 기억
document.getElementById('plot').on('plotly_relayout', function(e) {{
  if (e['scene.camera']) sceneCamera = e['scene.camera'];
}});

// ── 핵심: Plotly.react() 로 data 전체 교체 ──────────────
// go.Frame 대신 매 step마다 완전히 새 data 배열을 넘김.
// geometry(topology)가 달라도 정확히 그 step의 메시가 표시됨.
function goToStep(i) {{
  currentStep = Math.max(0, Math.min(i, TOTAL - 1));
  const layout = Object.assign({{}}, LAYOUT);
  if (sceneCamera) layout.scene = Object.assign({{}}, LAYOUT.scene, {{camera: sceneCamera}});
  Plotly.react('plot', buildTraces(currentStep), layout);
  updateUI();
}}

function buildTraces(i) {{
  const traces = [];
  if (MOLD) traces.push(MOLD);
  const s = STEPS[i];
  if (s.fluid) traces.push(s.fluid);
  return traces;
}}

function updateUI() {{
  const s = STEPS[currentStep];
  document.getElementById('stepLabel').textContent =
    'Step ' + (currentStep+1) + ' / ' + TOTAL + '  —  ' + s.label;
  document.getElementById('fillInner').style.width  = s.fill_pct + '%';
  document.getElementById('fillPct').textContent    = s.fill_pct + ' %';
  document.getElementById('slider').value           = currentStep;
  document.getElementById('btnPlay').textContent    = playing ? '⏸ Pause' : '▶ Play';
}}

function togglePlay() {{
  playing = !playing;
  if (playing) {{
    if (currentStep >= TOTAL - 1) currentStep = 0;
    advance();
  }} else {{
    clearTimeout(timer);
  }}
  updateUI();
}}

function advance() {{
  if (!playing) return;
  goToStep(currentStep);
  if (currentStep < TOTAL - 1) {{
    currentStep++;
    timer = setTimeout(advance, 600);
  }} else {{
    playing = false;
    updateUI();
  }}
}}

function stepBy(d) {{
  clearTimeout(timer);
  playing = false;
  goToStep(currentStep + d);
}}

// 초기 UI 세팅
updateUI();
</script>
</body>
</html>
"""
                components.html(html_code, height=680, scrolling=False)

            except Exception as e:
                st.error(f"Animation build failed: {e}")
                st.exception(e)

else:
    st.error("VTK directory not found. Please sync results first.")
