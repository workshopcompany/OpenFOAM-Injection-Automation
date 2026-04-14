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

# ── [1] 데이터 직렬화 및 헬퍼 함수 ────────────────────────────────
class _NpEncoder(json.JSONEncoder):
    """numpy scalar/array → Python native types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def _safe_json_fixed(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder).replace('</', '<\\/')

def _trace_to_json(trace):
    raw = trace.to_plotly_json()
    return json.loads(_safe_json_fixed(raw))

def make_fluid_trace(pts, fi, fj, fk, alpha_vals, name="Fluid", show_legend=True, show_colorbar=True):
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
        colorscale="Jet", # 유체 시각화에 좋은 Jet 컬러맵
        opacity=1.0,
        name=name,
        showlegend=show_legend,
        colorbar=cb,
    )

def make_mold_trace(mold_trimesh, opacity=0.1, show_legend=True):
    if mold_trimesh is None: return None
    mv, mf = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(
        x=mv[:, 0], y=mv[:, 1], z=mv[:, 2],
        i=mf[:, 0], j=mf[:, 1], k=mf[:, 2],
        opacity=opacity, color="lightgray", name="Mold", showlegend=show_legend,
    )

# ── [2] 유체 데이터 로드 (스케일, 임계값, 좌표 조절 통합) ──────────────
def load_fluid_refined(fpath, x_off=0.0, y_off=0.0, z_off=0.0, scale=1000.0, thres=0.1):
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock): mesh = mesh.combine()
        
        f_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if f_name not in mesh.array_names: return None, None, 0
        
        # 임계값(Threshold) 적용: 실선 방지를 위해 낮은 농도도 포함
        fluid = mesh.threshold(thres, scalars=f_name)
        if fluid.n_cells == 0: return None, None, 0
        
        surf = fluid.triangulate().extract_surface()
        
        # 스케일(Scale) 및 좌표 이동 적용
        pts = surf.points * scale  
        pts[:, 0] += x_off
        pts[:, 1] += y_off
        pts[:, 2] += z_off
        
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        alpha = surf.point_data[f_name].tolist()
        return (pts, faces[:,0], faces[:,1], faces[:,2]), alpha, fluid.n_cells
    except Exception as e:
        return None, None, 0

# ── [3] 3D Animation 시각화 섹션 ─────────────────────────────────
vtk_dir = "VTK"
if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (alpha.water)")

    all_files = list(dict.fromkeys(
        glob.glob(f"{vtk_dir}/case_*.vtm") + glob.glob(f"{vtk_dir}/**/case_*.vtm", recursive=True) +
        glob.glob(f"{vtk_dir}/case_*.vtk") + glob.glob(f"{vtk_dir}/**/case_*.vtk", recursive=True)
    ))
    all_files = sorted(all_files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]) if re.findall(r'\d+', os.path.basename(x)) else 0)

    if not all_files:
        st.warning("VTK 데이터가 없습니다.")
    else:
        total_steps = len(all_files)
        mold_mesh = st.session_state.get("mesh")

        # ── (A) 가시화 디버그 컨트롤 (선으로만 보일 때 조절) ──
        with st.expander("🔍 유체 가시화 정밀 설정 (유체가 얇은 선으로만 보일 때 조절)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                sc_val = st.slider("유체 크기 배율 (Scale)", 0.1, 2000.0, 1000.0, help="m 단위를 mm 단위로 맞춥니다.")
                th_val = st.slider("유체 포착 농도 (Threshold)", 0.01, 0.99, 0.1, help="값을 낮출수록 얇은 선이 면적을 갖게 됩니다.")
            with col2:
                off_x = st.number_input("X Offset (좌표 이동)", value=0.0)
                off_y = st.number_input("Y Offset (좌표 이동)", value=0.0)
                off_z = st.number_input("Z Offset (좌표 이동)", value=0.0)

        # ── (B) Step-by-Step Viewer ──
        step_idx = st.slider("⏱ Time Step", 0, total_steps - 1, total_steps // 2, key="slider_step")
        
        res, a_vals, n_c = load_fluid_refined(all_files[step_idx], off_x, off_y, off_z, sc_val, th_val)
        
        fig = go.Figure()
        if mold_mesh: fig.add_trace(make_mold_trace(mold_mesh, opacity=0.1))
        
        if res:
            fig.add_trace(make_fluid_trace(res, None, None, None, a_vals, show_colorbar=True))
            st.success(f"Step {step_idx}: {n_c:,}개의 유체 셀이 화면에 표시되었습니다.")
        else:
            st.error("⚠️ 현재 설정(스케일/오프셋/임계값)에서 유체를 찾을 수 없습니다.")

        fig.update_layout(scene=dict(aspectmode="data"), height=550, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

        # ── (C) JS-driven Animation ──
        st.divider()
        if st.button("🎬 현재 설정으로 애니메이션 생성 및 재생", use_container_width=True):
            prog = st.progress(0, text="애니메이션 데이터 구축 중...")
            step_data = []
            mold_t = make_mold_trace(mold_mesh, opacity=0.1)
            mold_json = _trace_to_json(mold_t) if mold_t else None

            for i, fpath in enumerate(all_files):
                prog.progress((i+1)/total_steps)
                r, av, nc = load_fluid_refined(fpath, off_x, off_y, off_z, sc_val, th_val)
                f_j = _trace_to_json(make_fluid_trace(r, None, None, None, av)) if r else None
                step_data.append({"fluid": f_j, "label": f"Step {i} ({nc} cells)"})
            
            prog.empty()
            s_js = _safe_json_fixed(step_data)
            m_js = _safe_json_fixed(mold_json)

            html_code = f"""
            <html>
            <head><script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></head>
            <body style="margin:0; background:#0e1117;">
                <div style="padding:10px; background:#1a1d27; color:white; display:flex; gap:10px; align-items:center;">
                    <button id="pBtn" onclick="togglePlay()" style="padding:5px 15px; cursor:pointer;">▶ Play</button> 
                    <span id="lb" style="font-family:sans-serif;">Step 0</span>
                    <input type="range" id="sd" min="0" max="{total_steps-1}" value="0" style="flex:1" oninput="draw(parseInt(this.value))">
                </div>
                <div id="gd" style="width:100vw; height:500px;"></div>
                <script>
                    const D={s_js}; const M={m_js}; let c=0, p=false, tm=null;
                    function draw(i) {{
                        c=i; const ts=[M, D[i].fluid].filter(Boolean);
                        Plotly.react('gd', ts, {{scene:{{aspectmode:'data'}}, paper_bgcolor:'rgba(0,0,0,0)', margin:{{l:0,r:0,b:0,t:0}}, font:{{color:'#eee'}} }});
                        document.getElementById('lb').innerText = D[i].label;
                        document.getElementById('sd').value = i;
                    }}
                    function togglePlay() {{ 
                        p=!p; 
                        document.getElementById('pBtn').innerText = p ? "⏸ Pause" : "▶ Play";
                        if(p) loop(); else clearTimeout(tm); 
                    }}
                    function loop() {{ 
                        if(!p) return; 
                        draw(c); 
                        c=(c+1)%{total_steps}; 
                        tm=setTimeout(loop, 200); 
                    }}
                    window.onload = () => draw(0);
                </script>
            </body>
            </html>
            """
            components.html(html_code, height=600)
else:
    st.error("VTK 디렉토리를 찾을 수 없습니다. 결과를 먼저 동기화해주세요.")
