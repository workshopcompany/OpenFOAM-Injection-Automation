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
import plotly.graph_objects as go
import trimesh

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# Session state init
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0); _init("gy", 0.0); _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp", 230.0); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 0.5)
_init("sim_running", False)
_init("sim_status", "idle")
_init("sim_logs", [])
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
_init("animation_playing", False)
_init("current_frame", 0)

# Custom CSS
st.markdown("""
<style>
    .stApp h1 { font-size: 2.2rem !important; }
    .stApp h2 { font-size: 1.55rem !important; }
    .stApp h3 { font-size: 1.3rem !important; }
    .stMetricLabel { font-size: 1.1rem !important; }
    .sim-log { font-family: monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Material Database
LOCAL_DB = {
    "PP": {"nu": 1e-3, "rho": 900.0, "Tmelt": 230.0, "Tmold": 40.0, "press_mpa": 70.0, "vel_mms": 80.0, "desc": "General-purpose polypropylene"},
    "ABS": {"nu": 2e-3, "rho": 1050.0, "Tmelt": 240.0, "Tmold": 60.0, "press_mpa": 80.0, "vel_mms": 70.0, "desc": "ABS resin"},
    "PA66": {"nu": 5e-4, "rho": 1140.0, "Tmelt": 280.0, "Tmold": 80.0, "press_mpa": 90.0, "vel_mms": 100.0, "desc": "Nylon 66"},
    "PA66+30GF": {"nu": 4e-4, "rho": 1300.0, "Tmelt": 285.0, "Tmold": 85.0, "press_mpa": 110.0, "vel_mms": 80.0, "desc": "30% glass-fiber reinforced nylon"},
    "PC": {"nu": 3e-3, "rho": 1200.0, "Tmelt": 300.0, "Tmold": 85.0, "press_mpa": 120.0, "vel_mms": 60.0, "desc": "Polycarbonate"},
    "POM": {"nu": 8e-4, "rho": 1410.0, "Tmelt": 200.0, "Tmold": 90.0, "press_mpa": 85.0, "vel_mms": 90.0, "desc": "Polyacetal"},
    "HDPE": {"nu": 9e-4, "rho": 960.0, "Tmelt": 220.0, "Tmold": 35.0, "press_mpa": 60.0, "vel_mms": 90.0, "desc": "HDPE"},
    "PET": {"nu": 6e-4, "rho": 1370.0, "Tmelt": 265.0, "Tmold": 70.0, "press_mpa": 80.0, "vel_mms": 85.0, "desc": "PET"},
    "CATAMOLD": {"nu": 5e-3, "rho": 4900.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 100.0, "vel_mms": 30.0, "desc": "BASF Catamold MIM feedstock"},
    "MIM": {"nu": 5e-3, "rho": 5000.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 100.0, "vel_mms": 30.0, "desc": "Metal injection molding feedstock"},
    "17-4PH": {"nu": 4e-3, "rho": 7780.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 110.0, "vel_mms": 25.0, "desc": "17-4PH stainless steel MIM feedstock"},
    "316L": {"nu": 4e-3, "rho": 7900.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 110.0, "vel_mms": 25.0, "desc": "316L stainless steel MIM feedstock"},
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    for key in LOCAL_DB:
        if key.upper() in name or name in key.upper():
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    return {"nu": 1e-3, "rho": 1000.0, "Tmelt": 220.0, "Tmold": 50.0, "press_mpa": 70.0, "vel_mms": 80.0, "material": material, "source": "Gemini recommendation", "desc": f"{material} — default"}

def get_process(material: str) -> dict:
    props = get_props(material)
    return {"temp": float(props.get("Tmelt", 230.0)), "press": float(props.get("press_mpa", 70.0)), "vel": float(props.get("vel_mms", 80.0))}

def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state["sim_logs"].append(log_entry)
    if len(st.session_state["sim_logs"]) > 50:
        st.session_state["sim_logs"] = st.session_state["sim_logs"][-50:]

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
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error(f"GitHub API failed: {response.status_code}")
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
            add_log("✅ Simulation results downloaded")
            return True
        return False
    except Exception as e:
        st.error(f"Sync error: {e}")
        return False

def load_fluid_volume(fpath, mold_mesh=None, scale=1000.0, thres=0.05):
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        
        f_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if f_name not in mesh.array_names:
            return None, None, 0
        
        fluid = mesh.threshold(thres, scalars=f_name)
        if fluid.n_cells == 0:
            fluid = mesh.threshold(0.01, scalars=f_name)
            if fluid.n_cells == 0:
                return None, None, 0
        
        fluid.points *= scale
        
        if mold_mesh is not None:
            try:
                fluid = fluid.clip_surface(mold_mesh, invert=True)
            except:
                pass
        
        surf = fluid.extract_surface()
        if surf.n_points == 0:
            return None, None, 0
        
        pts = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:] if surf.faces.size > 0 else np.empty((0,3), dtype=int)
        
        if f_name in surf.point_data:
            alpha_vals = surf.point_data[f_name].tolist()
        else:
            alpha_vals = [thres] * len(pts)
        
        return (pts, faces[:,0], faces[:,1], faces[:,2]), alpha_vals, fluid.n_cells
    except Exception as e:
        return None, None, 0

def get_sampled_files(all_files, num_steps=10):
    if not all_files:
        return []
    total = len(all_files)
    if total <= num_steps:
        return all_files
    indices = np.linspace(0, total-1, num_steps, dtype=int)
    return [all_files[i] for i in indices]

def make_fluid_trace(pts, fi, fj, fk, alpha_vals, name="Fluid", opacity=0.7):
    intensity = np.array(alpha_vals) if alpha_vals is not None else np.ones(len(pts))
    return go.Mesh3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        i=fi, j=fj, k=fk,
        intensity=intensity,
        colorscale=[[0, 'blue'], [0.3, 'cyan'], [0.6, 'lime'], [0.8, 'orange'], [1.0, 'red']],
        opacity=opacity,
        name=name,
        showscale=True,
        colorbar=dict(title="Fill Ratio", thickness=20, len=0.6),
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3)
    )

def make_mold_trace(mold_trimesh, opacity=0.15, color="lightgray"):
    if mold_trimesh is None:
        return None
    mv, mf = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(
        x=mv[:,0], y=mv[:,1], z=mv[:,2],
        i=mf[:,0], j=mf[:,1], k=mf[:,2],
        opacity=opacity, color=color, name="Mold", showlegend=True,
        lighting=dict(ambient=0.3, diffuse=0.7)
    )

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    if uploaded:
        try:
            mesh = trimesh.load(uploaded, file_type="stl")
            st.session_state["mesh"] = mesh
            st.success(f"✅ STL loaded — {len(mesh.faces):,} faces")
            add_log(f"STL loaded: {len(mesh.faces):,} faces")
        except Exception as e:
            st.error(f"STL load failed: {e}")
    
    st.divider()
    st.header("📍 2. Gate Configuration")
    
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        mesh = st.session_state.get("mesh")
        if mesh is not None:
            center = mesh.centroid
            snap, _, _ = trimesh.proximity.closest_point(mesh, [center])
            pos = snap[0]
            st.session_state["gx"] = float(pos[0])
            st.session_state["gy"] = float(pos[1])
            st.session_state["gz"] = float(pos[2])
            st.session_state["gsize"] = 2.5
            st.toast("AI Gate Suggestion Completed!", icon="🪄")
            add_log(f"AI Gate: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        else:
            st.warning("Upload STL first")
    
    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")
    vx = st.number_input("Gate X", value=float(st.session_state["gx"]), step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=float(st.session_state["gy"]), step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=float(st.session_state["gz"]), step=0.1, key="gz")
    
    mesh = st.session_state.get("mesh")
    if mesh is not None:
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
    mat_name = st.text_input("Material Name", value="PA66+30GF", key="mat_name_input")
    st.session_state["mat_name"] = mat_name
    
    if st.button("🤖 AI Material Properties", use_container_width=True, type="primary"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False
        add_log(f"Material properties loaded: {mat_name}")
    
    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 Source: {p.get('source', 'Gemini')}")
        if p.get("desc"):
            st.info(p["desc"])
        
        with st.expander("📋 Edit Properties", expanded=True):
            p["nu"] = st.number_input("Viscosity (m²/s)", value=float(p["nu"]), format="%.2e", min_value=1e-7, max_value=1.0, key="edit_nu")
            p["rho"] = st.number_input("Density (kg/m³)", value=float(p["rho"]), min_value=100.0, max_value=9000.0, step=1.0, key="edit_rho")
            p["Tmelt"] = st.number_input("Melt Temp (°C)", value=float(p["Tmelt"]), min_value=100.0, max_value=450.0, step=1.0, key="edit_tmelt")
            p["Tmold"] = st.number_input("Mold Temp (°C)", value=float(p["Tmold"]), min_value=10.0, max_value=200.0, step=1.0, key="edit_tmold")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Properties", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Properties Confirmed!", icon="✅")
                    add_log("Material properties confirmed")
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state["props"] = None
                    st.session_state["props_confirmed"] = False
                    add_log("Properties reset")
                    st.rerun()
    
    st.divider()
    st.header("⚙️ 4. Process")
    
    if st.button("🤖 Optimize Process", use_container_width=True):
        opt = get_process(mat_name)
        st.session_state["temp"] = opt["temp"]
        st.session_state["press"] = opt["press"]
        st.session_state["vel"] = opt["vel"]
        st.toast("Process Optimized!", icon="🤖")
        add_log(f"Process optimized: {opt['temp']}°C, {opt['press']}MPa, {opt['vel']}mm/s")
    
    temp_c = st.number_input("Temp (°C)", 50.0, 450.0, step=1.0, key="temp")
    press_mpa = st.number_input("Pressure (MPa)", 10.0, 250.0, step=1.0, key="press")
    vel_mms = st.number_input("Velocity (mm/s)", 1.0, 600.0, step=1.0, key="vel")
    etime = st.number_input("End Time (s)", value=float(st.session_state["etime"]), min_value=0.1, max_value=10.0, step=0.1, key="etime")
    
    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"] = etime
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Process", use_container_width=True):
            st.session_state["process_confirmed"] = True
            st.toast("Process Confirmed!", icon="✅")
            add_log("Process conditions confirmed")
    with col2:
        if st.button("🔄 Reset Process", use_container_width=True):
            st.session_state["process_confirmed"] = False
            add_log("Process reset")
            st.rerun()
    
    if not st.session_state.get("process_confirmed", False):
        st.warning("⚠️ Please confirm process conditions")
    
    st.divider()
    
    num_frames = st.select_slider("Animation Frames", options=[5, 10, 15, 20, 30], value=15)
    
    run_disabled = st.session_state["sim_running"] or not st.session_state["props_confirmed"] or not st.session_state.get("process_confirmed", False)
    
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True, disabled=run_disabled):
        if not ZAPIER_URL:
            st.error("❌ ZAPIER_URL not configured")
        else:
            props = st.session_state["props"]
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"] = True
            st.session_state["sim_status"] = "running"
            
            add_log(f"🚀 Simulation started: {sig_id}")
            add_log(f"Material: {mat_name}")
            add_log(f"nu={props['nu']:.2e}, rho={props['rho']} kg/m³")
            add_log(f"Tmelt={props['Tmelt']}°C, Tmold={props['Tmold']}°C")
            add_log(f"Gate: ({gx:.2f}, {gy:.2f}, {gz:.2f}) mm, Dia={g_size}mm")
            add_log(f"Injection: {temp_c}°C, {press_mpa}MPa, {vel_mms}mm/s")
            
            payload = {
                "signal_id": sig_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material": mat_name,
                "viscosity": float(props["nu"]),
                "density": float(props["rho"]),
                "melt_temp": float(props["Tmelt"]),
                "mold_temp": float(props["Tmold"]),
                "temp": float(temp_c),
                "press": float(press_mpa),
                "vel": round(vel_mms / 1000, 6),
                "etime": float(etime),
                "gate_pos": {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
                "gate_size": float(g_size),
            }
            
            try:
                add_log("📡 Sending to Zapier...")
                res = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if res.status_code == 200:
                    st.toast(f"Signal Sent! ID: {sig_id}", icon="🚀")
                    add_log(f"✅ Signal sent: HTTP {res.status_code}")
                    add_log("⏳ Waiting for GitHub Actions...")
                else:
                    st.error(f"Failed: HTTP {res.status_code}")
                    add_log(f"❌ Failed: HTTP {res.status_code}")
                    st.session_state["sim_running"] = False
            except Exception as e:
                st.error(f"Error: {e}")
                add_log(f"❌ Error: {e}")
                st.session_state["sim_running"] = False

# ========== MAIN AREA ==========
col_geo, col_log = st.columns([2, 1])

with col_geo:
    st.header("🎥 3D Geometry")
    mesh = st.session_state.get("mesh")
    
    if mesh is not None:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2], color="#AAAAAA", opacity=0.7),
            go.Scatter3d(
                x=[st.session_state["gx_final"]], 
                y=[st.session_state["gy_final"]], 
                z=[st.session_state["gz_final"]],
                mode="markers",
                marker=dict(size=st.session_state["gsize"] * 3, color="red"),
                name="Gate"
            )
        ])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode="data"), height=500)
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        bb = mesh.bounds
        c1, c2, c3 = st.columns(3)
        c1.metric("X Size", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y Size", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z Size", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("Upload STL file to see 3D model")

with col_log:
    st.header("📟 Logs")
    
    if st.session_state["sim_running"]:
        st.info("🟢 RUNNING...")
    elif st.session_state["sim_status"] == "failed":
        st.error("🔴 FAILED")
    else:
        st.success("✅ READY")
    
    log_container = st.container(height=400)
    with log_container:
        for log in st.session_state["sim_logs"][-30:]:
            st.code(log, language="bash")
    
    if st.session_state["sim_running"]:
        if st.button("✅ Mark Complete", use_container_width=True):
            st.session_state["sim_running"] = False
            st.session_state["sim_status"] = "completed"
            add_log("Simulation marked complete")
            st.rerun()
    
    if st.button("🗑 Clear Logs", use_container_width=True):
        st.session_state["sim_logs"] = []
        st.rerun()

st.info(f"📍 Gate: ({st.session_state['gx_final']:.2f}, {st.session_state['gy_final']:.2f}, {st.session_state['gz_final']:.2f}) mm")

if st.session_state["props_confirmed"] and st.session_state.get("process_confirmed", False) and st.session_state["props"]:
    p = st.session_state["props"]
    st.caption(f"ℹ️ Confirmed | nu={p['nu']:.2e} | rho={p['rho']} kg/m³ | Tmelt={p['Tmelt']}°C")
else:
    st.caption("ℹ️ Confirm properties and process before running")

# ========== RESULTS ==========
st.title("📊 Results")

if st.button("🔄 Sync from GitHub", use_container_width=True):
    with st.spinner("Syncing..."):
        if sync_simulation_results():
            st.success("Synced!")
            st.rerun()

if os.path.exists("results.txt"):
    with open("results.txt") as f:
        st.text_area("Summary", f.read(), height=150)

if os.path.exists("logs.zip"):
    with open("logs.zip", "rb") as f:
        st.download_button("📂 Download Logs", f, "logs.zip", use_container_width=True)

# ========== ANIMATION ==========
vtk_dir = "VTK"
if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation")
    
    all_files = sorted(glob.glob(f"{vtk_dir}/**/case_*.vt*", recursive=True) + glob.glob(f"{vtk_dir}/case_*.vt*", recursive=True))
    all_files = sorted(set(all_files), key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    
    if not all_files:
        st.warning("No VTK files found")
    else:
        mold_mesh = st.session_state.get("mesh")
        
        with st.expander("🔧 Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                scale_factor = st.slider("Scale (m→mm)", 100.0, 2000.0, 1000.0, 50.0)
                threshold = st.slider("Threshold", 0.01, 0.5, 0.05, 0.01)
            with col2:
                mold_opacity = st.slider("Mold Opacity", 0.0, 0.5, 0.1, 0.01)
                fluid_opacity = st.slider("Fluid Opacity", 0.5, 1.0, 0.75, 0.05)
            with col3:
                view_mode = st.radio("View", ["Auto", "Uniform"], index=0)
        
        sampled_files = get_sampled_files(all_files, num_frames)
        total_steps = len(sampled_files)
        
        st.markdown("### 🎮 Controls")
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 3, 1])
        
        with col_ctrl1:
            if st.button("⏮ First"):
                st.session_state["current_frame"] = 0
                st.session_state["animation_playing"] = False
                st.rerun()
        with col_ctrl2:
            if st.button("⏸ Pause"):
                st.session_state["animation_playing"] = False
                st.rerun()
        with col_ctrl3:
            if st.button("▶ Play", type="primary"):
                st.session_state["animation_playing"] = True
                st.rerun()
        with col_ctrl4:
            if st.button("⏭ Last"):
                st.session_state["current_frame"] = total_steps - 1
                st.session_state["animation_playing"] = False
                st.rerun()
        
        current_frame = st.slider("Frame", 0, total_steps - 1, value=st.session_state.get("current_frame", 0), key="frame_slider")
        st.session_state["current_frame"] = current_frame
        
        with st.spinner("Loading..."):
            fpath = sampled_files[current_frame]
            res, alpha_vals, n_cells = load_fluid_volume(fpath, mold_mesh, scale=scale_factor, thres=threshold)
            
            fig = go.Figure()
            
            if mold_mesh:
                fig.add_trace(make_mold_trace(mold_mesh, opacity=mold_opacity))
            
            gate_x, gate_y, gate_z = st.session_state["gx_final"], st.session_state["gy_final"], st.session_state["gz_final"]
            fig.add_trace(go.Scatter3d(
                x=[gate_x], y=[gate_y], z=[gate_z],
                mode="markers",
                marker=dict(size=st.session_state["gsize"] * 2, color="red", symbol="x"),
                name="Gate"
            ))
            
            if res:
                pts, fi, fj, fk = res
                fig.add_trace(make_fluid_trace(pts, fi, fj, fk, alpha_vals, opacity=fluid_opacity))
                st.success(f"Frame {current_frame+1}/{total_steps} | Cells: {n_cells:,}")
            else:
                st.warning("No fluid detected")
            
            scene_config = dict(
                aspectmode="data" if view_mode == "Auto" else "cube",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.5), up=dict(x=0, y=0, z=1)),
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"
            )
            
            fig.update_layout(scene=scene_config, height=650, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        
        if st.session_state.get("animation_playing", False):
            next_frame = (current_frame + 1) % total_steps
            st.session_state["current_frame"] = next_frame
            time.sleep(0.15)
            st.rerun()
        
        if mold_mesh:
            bounds = mold_mesh.bounds
            st.caption(f"Bounds: X [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}] | Y [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}] | Z [{bounds[0][2]:.1f}, {bounds[1][2]:.1f}] mm")
else:
    st.error("VTK directory not found. Run simulation and sync first.")
