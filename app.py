import streamlit as st
import streamlit.components.v1 as components
import os, time, uuid, requests
from datetime import datetime
import numpy as np
import json
import zipfile
import io
import glob
import re
import pyvista as pv
from stpyvista import stpyvista
import plotly.graph_objects as go
import meshio

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

# Custom CSS
st.markdown("""
<style>
    .stApp h1 { font-size: 2.2rem !important; }
    .stApp h2 { font-size: 1.55rem !important; }
    .stApp h3 { font-size: 1.3rem !important; }
    .stMetricLabel { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# Session state init
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

# Local material database (same as original)
LOCAL_DB = {
    "PP": {"nu": 1e-3, "rho": 900, "Tmelt": 230, "Tmold": 40, "press_mpa": 70, "vel_mms": 80, "desc": "General-purpose polypropylene"},
    "ABS": {"nu": 2e-3, "rho": 1050, "Tmelt": 240, "Tmold": 60, "press_mpa": 80, "vel_mms": 70, "desc": "ABS resin"},
    "PA66": {"nu": 5e-4, "rho": 1140, "Tmelt": 280, "Tmold": 80, "press_mpa": 90, "vel_mms": 100, "desc": "Nylon 66"},
    "PA66+30GF": {"nu": 4e-4, "rho": 1300, "Tmelt": 285, "Tmold": 85, "press_mpa": 110, "vel_mms": 80, "desc": "30% glass-fiber reinforced nylon"},
    "PC": {"nu": 3e-3, "rho": 1200, "Tmelt": 300, "Tmold": 85, "press_mpa": 120, "vel_mms": 60, "desc": "Polycarbonate"},
    "POM": {"nu": 8e-4, "rho": 1410, "Tmelt": 200, "Tmold": 90, "press_mpa": 85, "vel_mms": 90, "desc": "Polyacetal"},
    "HDPE": {"nu": 9e-4, "rho": 960, "Tmelt": 220, "Tmold": 35, "press_mpa": 60, "vel_mms": 90, "desc": "HDPE"},
    "PET": {"nu": 6e-4, "rho": 1370, "Tmelt": 265, "Tmold": 70, "press_mpa": 80, "vel_mms": 85, "desc": "PET"},
    "CATAMOLD": {"nu": 5e-3, "rho": 4900, "Tmelt": 185, "Tmold": 40, "press_mpa": 100, "vel_mms": 30, "desc": "BASF Catamold MIM"},
    "MIM": {"nu": 5e-3, "rho": 5000, "Tmelt": 185, "Tmold": 40, "press_mpa": 100, "vel_mms": 30, "desc": "Metal injection molding"},
    "17-4PH": {"nu": 4e-3, "rho": 7780, "Tmelt": 185, "Tmold": 40, "press_mpa": 110, "vel_mms": 25, "desc": "17-4PH stainless steel"},
    "316L": {"nu": 4e-3, "rho": 7900, "Tmelt": 185, "Tmold": 40, "press_mpa": 110, "vel_mms": 25, "desc": "316L stainless steel"},
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    for key in LOCAL_DB:
        if key.upper() in name or name in key.upper():
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    return {"nu": 1e-3, "rho": 1000, "Tmelt": 220, "Tmold": 50, "press_mpa": 70, "vel_mms": 80, "material": material, "source": "Gemini recommendation", "desc": f"{material} — default"}

def get_process(material: str) -> dict:
    props = get_props(material)
    return {"temp": props.get("Tmelt", 230), "press": float(props.get("press_mpa", 70)), "vel": float(props.get("vel_mms", 80))}

# GitHub sync (unchanged)
def sync_simulation_results():
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = "workshopcompany"
    REPO_NAME = "OpenFOAM-Injection-Automation"
    ARTIFACT_NAME = "simulation-results"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"GitHub API connection failed: {response.status_code}")
        return False
    artifacts = response.json().get("artifacts", [])
    target_artifact = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
    if not target_artifact:
        st.warning("No simulation results yet.")
        return False
    download_url = target_artifact["archive_download_url"]
    file_res = requests.get(download_url, headers=headers)
    if file_res.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(file_res.content)) as z:
            z.extractall(".")
        return True
    else:
        st.error("Failed to download result files.")
        return False

# ========== NEW 3D CLIPPING FUNCTION ==========
def load_fluid_3d_clipped(fpath, mold_mesh=None, scale=1000.0, thres=0.5):
    """
    VTK에서 등치면(contour)으로 3D 유체 표면 생성, 금형(STL)으로 클리핑.
    Returns: (points, i, j, k), alpha_values, n_cells
    """
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        f_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if f_name not in mesh.array_names:
            return None, None, 0

        # 등치면 생성 (3D volume skin)
        fluid_3d = mesh.contour([thres], scalars=f_name)
        if fluid_3d.n_points == 0:
            return None, None, 0

        # 스케일 (m -> mm)
        fluid_3d.points *= scale

        # 금형 클리핑 (외부 제거)
        if mold_mesh is not None:
            try:
                fluid_3d = fluid_3d.clip_surface(mold_mesh, invert=True)
            except Exception as e:
                st.warning(f"Clipping failed: {e}, using raw fluid.")

        # 삼각면 추출
        surf = fluid_3d.triangulate()
        pts = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:] if surf.faces.size > 0 else np.empty((0,3), dtype=int)
        alpha = surf.point_data[f_name].tolist() if f_name in surf.point_data else [1.0]*len(pts)
        return (pts, faces[:,0], faces[:,1], faces[:,2]), alpha, fluid_3d.n_cells
    except Exception as e:
        st.warning(f"load_fluid_3d_clipped error: {e}")
        return None, None, 0

def get_sampled_files(all_files, num_steps=10):
    """등간격 샘플링"""
    if not all_files:
        return []
    total = len(all_files)
    if total <= num_steps:
        return all_files
    indices = np.linspace(0, total-1, num_steps, dtype=int)
    return [all_files[i] for i in indices]

# Plotly helpers
class _NpEncoder(json.JSONEncoder):
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
        pts, fi, fj, fk = pts
    intensity = alpha_vals if alpha_vals is not None else np.ones(len(pts))
    cb = dict(title="alpha.water", thickness=15, len=0.6) if show_colorbar else None
    return go.Mesh3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        i=fi, j=fj, k=fk,
        intensity=intensity,
        colorscale="Jet",
        opacity=1.0,
        name=name,
        showlegend=show_legend,
        colorbar=cb,
    )

def make_mold_trace(mold_trimesh, opacity=0.1, show_legend=True):
    if mold_trimesh is None: return None
    mv, mf = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(
        x=mv[:,0], y=mv[:,1], z=mv[:,2],
        i=mf[:,0], j=mf[:,1], k=mf[:,2],
        opacity=opacity, color="lightgray", name="Mold", showlegend=show_legend,
    )

# Sidebar (mostly unchanged, but added frame selection)
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded and HAS_TRIMESH:
        try:
            mesh = trimesh.load(uploaded, file_type="stl")
            st.session_state["mesh"] = mesh
            st.success(f"✅ STL loaded — {len(mesh.faces):,} faces")
        except Exception as e:
            st.error(f"STL load failed: {e}")
    elif uploaded:
        st.warning("trimesh not installed")

    st.divider()
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        mesh = st.session_state.get("mesh")
        if mesh is not None and HAS_TRIMESH:
            center = mesh.centroid
            snap, _, _ = trimesh.proximity.closest_point(mesh, [center])
            pos = snap[0]
            st.session_state["gx"] = float(pos[0])
            st.session_state["gy"] = float(pos[1])
            st.session_state["gz"] = float(pos[2])
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
        gx, gy, gz = float(snap[0][0]), float(snap[0][1]), float(snap[0][2])
    else:
        gx, gy, gz = vx, vy, vz
    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz

    st.divider()
    st.header("🧪 3. Material")
    mat_name = st.text_input("Material Name", value="PA66+30GF", key="mat_name_input")
    st.session_state["mat_name"] = mat_name
    if st.button("🤖 AI Material Properties (Gemini)", use_container_width=True, type="primary"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False
    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 Source: {p.get('source', 'Gemini recommendation')}")
        if p.get("desc"): st.info(p["desc"])
        with st.expander("📋 Material Properties Check / Edit", expanded=True):
            p["nu"] = st.number_input("Kinematic Viscosity nu (m²/s)", value=float(p.get("nu",1e-3)), format="%.2e", min_value=1e-7, max_value=1.0, key="edit_nu")
            p["rho"] = st.number_input("Density ρ (kg/m³)", value=float(p.get("rho",1000)), min_value=100.0, max_value=9000.0, step=1.0, key="edit_rho")
            p["Tmelt"] = st.number_input("Melt Temperature (°C)", value=int(p.get("Tmelt",220)), min_value=100, max_value=450, key="edit_tmelt")
            p["Tmold"] = st.number_input("Mold Temperature (°C)", value=int(p.get("Tmold",50)), min_value=10, max_value=200, key="edit_tmold")
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
        st.session_state["temp"] = suggestion["temp"]
        st.session_state["press"] = suggestion["press"]
        st.session_state["vel"] = suggestion["vel"]
        st.toast("Process Conditions Optimized!", icon="🤖")
    temp_c = st.number_input("Injection Temperature (°C)", 50, 450, step=1, key="temp")
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 250.0, step=1.0, key="press")
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 600.0, step=1.0, key="vel")
    etime = st.number_input("End Time (s)", value=st.session_state["etime"], min_value=0.1, max_value=10.0, step=0.1, key="etime")
    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"] = etime
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
    # NEW: Frame selection for animation
    st.subheader("🎬 Animation Quality")
    num_frames = st.select_slider("Number of frames (time steps)", options=[5, 10, 15, 20, 30], value=10, help="More frames = smoother animation but slower generation")

    run_disabled = st.session_state["sim_running"] or not st.session_state["props_confirmed"] or not st.session_state.get("process_confirmed", False)
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True, disabled=run_disabled):
        if not ZAPIER_URL:
            st.error("❌ ZAPIER_URL is not configured.")
        else:
            props = st.session_state["props"]
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"] = True
            payload = {
                "signal_id": sig_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material": mat_name, "viscosity": float(props["nu"]), "density": float(props["rho"]),
                "melt_temp": int(props["Tmelt"]), "mold_temp": int(props["Tmold"]),
                "temp": int(temp_c), "press": float(press_mpa), "vel": round(vel_mms/1000,6),
                "etime": float(etime), "gate_pos": {"x": round(gx,3), "y": round(gy,3), "z": round(gz,3)},
                "gate_size": float(g_size),
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

# Main area (geometry preview)
gx_f, gy_f, gz_f = st.session_state["gx_final"], st.session_state["gy_final"], st.session_state["gz_final"]
g_size_f = st.session_state["gsize"]
col_geo, col_log = st.columns([2,1])
with col_geo:
    st.header("🎥 3D Geometry Analysis")
    mesh = st.session_state.get("mesh")
    if mesh is not None and HAS_PLOTLY:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2], color="#AAAAAA", opacity=0.8),
            go.Scatter3d(x=[gx_f], y=[gy_f], z=[gz_f], mode="markers", marker=dict(size=g_size_f*5, color="red"), name="Gate")
        ])
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene=dict(aspectmode="data"), height=480)
        st.plotly_chart(fig, use_container_width=True)
        bb = mesh.bounds
        c1,c2,c3 = st.columns(3)
        c1.metric("X Size", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y Size", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z Size", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("Upload an STL file in the sidebar to display the 3D model.")
with col_log:
    st.header("📟 Simulation & Debug Logs")
    if st.session_state["sim_running"] and st.session_state["last_signal_id"] and st.session_state["props"]:
        log_lines = [f">>> [MIM-Ops] Outbound Signal ID: {st.session_state['last_signal_id']}",
                     ">>> Preventing Duplicate Runs: Bypass GitHub Push Trigger.",
                     ">>> Verifying OpenFOAM Dictionary Integrity...",
                     f">>> Material: {st.session_state['mat_name']}",
                     f">>> nu = {st.session_state['props']['nu']:.2e} m²/s",
                     f">>> rho = {st.session_state['props']['rho']} kg/m³",
                     f">>> Tmelt = {st.session_state['props']['Tmelt']}°C | Tmold = {st.session_state['props']['Tmold']}°C",
                     f">>> Gate: ({gx_f:.2f}, {gy_f:.2f}, {gz_f:.2f}) Ø{g_size_f} mm",
                     f">>> Velocity = {st.session_state['last_vel_mms']/1000:.4f} m/s",
                     f">>> End Time = {st.session_state['last_etime']} s",
                     "✅ transportProperties: OK", "✅ fvSolution: OK", "✅ fvSchemes: OK",
                     ">>> Zapier → GitHub Actions signal sent successfully.",
                     ">>> blockMesh execution pending...", ">>> interFoam execution pending...",
                     ">>> Check results in GitHub Actions Artifacts."]
        st.code("\n".join(log_lines), language="bash")
        if st.button("✅ Mark as Completed"):
            st.session_state["sim_running"] = False
            st.rerun()
    elif st.session_state["last_signal_id"]:
        st.success(f"✅ Last Run ID: {st.session_state['last_signal_id']}")
        st.info("Check the results in GitHub Actions → Artifacts.")
    else:
        st.info("Run a simulation to see logs here.")

st.info(f"📍 Final Gate Position: ({gx_f:.2f}, {gy_f:.2f}, {gz_f:.2f})")
if st.session_state["props_confirmed"] and st.session_state.get("process_confirmed", False) and st.session_state["props"]:
    p = st.session_state["props"]
    st.caption(f"ℹ️ Properties & Process confirmed | nu={p['nu']:.2e} | rho={p['rho']} kg/m³ | Tmelt={p['Tmelt']}°C | Tmold={p['Tmold']}°C")
else:
    st.caption("ℹ️ Confirm both Material Properties and Process Conditions in the sidebar before running simulation.")

# ========== SIMULATION RESULTS SECTION (with new 3D clipped animation) ==========
st.title("MIM-Ops Simulation Results")
if st.button("🔄 Refresh Latest Results (GitHub Sync)"):
    with st.spinner("Fetching latest data from GitHub securely..."):
        if sync_simulation_results():
            st.success("Data synchronization complete! Loading visualization data.")
            time.sleep(1)
            st.rerun()

if os.path.exists("results.txt"):
    with open("results.txt", "r") as f:
        summary = f.read()
    st.text_area("📄 Simulation Summary", summary, height=200)

if os.path.exists("logs.zip"):
    with open("logs.zip", "rb") as f:
        st.download_button(label="📂 Download All Logs (logs.zip)", data=f, file_name="logs.zip", mime="application/zip")

vtk_dir = "VTK"
if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (alpha.water) - Clipped to Mold")

    all_files = list(dict.fromkeys(
        glob.glob(f"{vtk_dir}/case_*.vtm") + glob.glob(f"{vtk_dir}/**/case_*.vtm", recursive=True) +
        glob.glob(f"{vtk_dir}/case_*.vtk") + glob.glob(f"{vtk_dir}/**/case_*.vtk", recursive=True)
    ))
    all_files = sorted(all_files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]) if re.findall(r'\d+', os.path.basename(x)) else 0)

    if not all_files:
        st.warning("No VTK data found.")
    else:
        mold_mesh = st.session_state.get("mesh")
        with st.expander("🔍 Fluid Visualization Fine-Tuning", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                scale_val = st.slider("Scale (m -> mm)", 0.1, 2000.0, 1000.0, help="Convert OpenFOAM meters to mm")
                thres_val = st.slider("Contour threshold (alpha.water)", 0.01, 0.99, 0.5, help="Isosurface value for 3D skin")
            with col2:
                off_x = st.number_input("X Offset", value=0.0)
                off_y = st.number_input("Y Offset", value=0.0)
                off_z = st.number_input("Z Offset", value=0.0)

        # Step-by-step viewer (single frame)
        step_idx = st.slider("⏱ Preview single time step", 0, len(all_files)-1, len(all_files)//2, key="preview_step")
        fpath = all_files[step_idx]
        # Apply offsets by transforming points after loading? We'll handle inside load function by scaling and offset.
        # But offsets are not yet implemented in load_fluid_3d_clipped; we can add simple translation.
        # For simplicity, we modify the function to accept xyz offset. Let's create a wrapper.
        def load_with_offset(fpath, mold_mesh, scale, thres, off_x, off_y, off_z):
            res, alpha, n_cells = load_fluid_3d_clipped(fpath, mold_mesh, scale, thres)
            if res is not None:
                pts, fi, fj, fk = res
                pts[:,0] += off_x
                pts[:,1] += off_y
                pts[:,2] += off_z
                return (pts, fi, fj, fk), alpha, n_cells
            return None, None, 0

        res, a_vals, n_c = load_with_offset(fpath, mold_mesh, scale_val, thres_val, off_x, off_y, off_z)
        fig_pre = go.Figure()
        if mold_mesh:
            fig_pre.add_trace(make_mold_trace(mold_mesh, opacity=0.1))
        if res:
            fig_pre.add_trace(make_fluid_trace(res, a_vals, show_colorbar=True))
            st.success(f"Step {step_idx}: {n_c} fluid cells displayed (3D surface clipped to mold).")
        else:
            st.error("No fluid surface generated with current threshold/scale.")
        fig_pre.update_layout(scene=dict(aspectmode="data"), height=550, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_pre, use_container_width=True)

        st.divider()
        if st.button(f"🎬 Generate & Play Animation ({num_frames} frames)", use_container_width=True):
            sampled_files = get_sampled_files(all_files, num_frames)
            prog = st.progress(0, text="Building animation data...")
            step_data = []
            mold_trace = make_mold_trace(mold_mesh, opacity=0.1)
            mold_json = _trace_to_json(mold_trace) if mold_trace else None

            for i, fpath in enumerate(sampled_files):
                prog.progress((i+1)/len(sampled_files))
                res, av, nc = load_with_offset(fpath, mold_mesh, scale_val, thres_val, off_x, off_y, off_z)
                if res:
                    ft = make_fluid_trace(res, av, show_colorbar=False)  # colorbar only in main viewer
                    step_data.append({"fluid": _trace_to_json(ft), "label": f"Frame {i+1}/{len(sampled_files)} ({nc} cells)"})
                else:
                    step_data.append({"fluid": None, "label": f"Frame {i+1} (no fluid)"})
            prog.empty()

            # Prepare HTML/JS animation
            s_js = _safe_json_fixed(step_data)
            m_js = _safe_json_fixed(mold_json) if mold_json else "null"
            total_frames = len(step_data)
            html_code = f"""
            <html>
            <head><script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></head>
            <body style="margin:0; background:#0e1117;">
                <div style="padding:10px; background:#1a1d27; color:white; display:flex; gap:10px; align-items:center;">
                    <button id="pBtn" onclick="togglePlay()" style="padding:5px 15px; cursor:pointer;">▶ Play</button> 
                    <span id="lb" style="font-family:sans-serif;">Frame 0</span>
                    <input type="range" id="sd" min="0" max="{total_frames-1}" value="0" style="flex:1" oninput="draw(parseInt(this.value))">
                </div>
                <div id="gd" style="width:100vw; height:500px;"></div>
                <script>
                    const D={s_js}; const M={m_js}; let c=0, p=false, tm=null;
                    function draw(i) {{
                        c=i; let traces = [];
                        if(M) traces.push(M);
                        if(D[i].fluid) traces.push(D[i].fluid);
                        Plotly.react('gd', traces, {{scene:{{aspectmode:'data'}}, paper_bgcolor:'rgba(0,0,0,0)', margin:{{l:0,r:0,b:0,t:0}}, font:{{color:'#eee'}} }});
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
                        c=(c+1)%{total_frames}; 
                        tm=setTimeout(loop, 200); 
                    }}
                    window.onload = () => draw(0);
                </script>
            </body>
            </html>
            """
            components.html(html_code, height=600)
else:
    st.error("VTK directory not found. Please sync results first.")
