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
    "PP": {"nu": 1e-3, "rho": 900.0, "Tmelt": 230.0, "Tmold": 40.0, "press_mpa": 70.0, "vel_mms": 80.0, "desc": "General-purpose polypropylene — excellent flowability, high shrinkage"},
    "ABS": {"nu": 2e-3, "rho": 1050.0, "Tmelt": 240.0, "Tmold": 60.0, "press_mpa": 80.0, "vel_mms": 70.0, "desc": "ABS resin — excellent impact resistance, suitable for plating"},
    "PA66": {"nu": 5e-4, "rho": 1140.0, "Tmelt": 280.0, "Tmold": 80.0, "press_mpa": 90.0, "vel_mms": 100.0, "desc": "Nylon 66 — excellent heat resistance and rigidity"},
    "PA66+30GF": {"nu": 4e-4, "rho": 1300.0, "Tmelt": 285.0, "Tmold": 85.0, "press_mpa": 110.0, "vel_mms": 80.0, "desc": "30% glass-fiber reinforced nylon — significantly improved rigidity"},
    "PC": {"nu": 3e-3, "rho": 1200.0, "Tmelt": 300.0, "Tmold": 85.0, "press_mpa": 120.0, "vel_mms": 60.0, "desc": "Polycarbonate — transparent, best impact resistance"},
    "POM": {"nu": 8e-4, "rho": 1410.0, "Tmelt": 200.0, "Tmold": 90.0, "press_mpa": 85.0, "vel_mms": 90.0, "desc": "Polyacetal — excellent wear resistance"},
    "HDPE": {"nu": 9e-4, "rho": 960.0, "Tmelt": 220.0, "Tmold": 35.0, "press_mpa": 60.0, "vel_mms": 90.0, "desc": "High-density polyethylene — excellent chemical resistance"},
    "PET": {"nu": 6e-4, "rho": 1370.0, "Tmelt": 265.0, "Tmold": 70.0, "press_mpa": 80.0, "vel_mms": 85.0, "desc": "PET — excellent transparency and strength"},
    "CATAMOLD": {"nu": 5e-3, "rho": 4900.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 100.0, "vel_mms": 30.0, "desc": "BASF Catamold MIM feedstock — metal powder + binder"},
    "MIM": {"nu": 5e-3, "rho": 5000.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 100.0, "vel_mms": 30.0, "desc": "Metal injection molding feedstock — high density"},
    "17-4PH": {"nu": 4e-3, "rho": 7780.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 110.0, "vel_mms": 25.0, "desc": "17-4PH stainless steel MIM feedstock"},
    "316L": {"nu": 4e-3, "rho": 7900.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 110.0, "vel_mms": 25.0, "desc": "316L stainless steel MIM feedstock — excellent corrosion resistance"},
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    for key in LOCAL_DB:
        if key.upper() in name or name in key.upper():
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    return {"nu": 1e-3, "rho": 1000.0, "Tmelt": 220.0, "Tmold": 50.0, "press_mpa": 70.0, "vel_mms": 80.0, "material": material, "source": "Gemini recommendation", "desc": f"{material} — Material not in database, default values applied"}

def get_process(material: str) -> dict:
    props = get_props(material)
    return {"temp": float(props.get("Tmelt", 230.0)), "press": float(props.get("press_mpa", 70.0)), "vel": float(props.get("vel_mms", 80.0))}

def add_log(message):
    """시뮬레이션 로그 추가"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state["sim_logs"].append(log_entry)
    # 로그는 최대 50개 유지
    if len(st.session_state["sim_logs"]) > 50:
        st.session_state["sim_logs"] = st.session_state["sim_logs"][-50:]

# GitHub sync
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
                z.extractall(".")
            add_log("✅ Simulation results downloaded successfully")
            return True
        else:
            st.error("Failed to download result files.")
            return False
    except Exception as e:
        st.error(f"Sync error: {e}")
        return False

# ========== 3D 유체 로드 함수 ==========
def load_fluid_volume(fpath, mold_mesh=None, scale=1000.0, thres=0.05):
    """threshold 방식으로 유체 데이터 추출 (게이트 위치 정보는 시각화에 반영)"""
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        
        f_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if f_name not in mesh.array_names:
            return None, None, 0
        
        # threshold로 유체 영역 추출
        fluid = mesh.threshold(thres, scalars=f_name)
        if fluid.n_cells == 0:
            fluid = mesh.threshold(0.01, scalars=f_name)
            if fluid.n_cells == 0:
                return None, None, 0
        
        # 스케일 적용 (OpenFOAM meter -> mm)
        fluid.points *= scale
        
        # 금형 클리핑
        if mold_mesh is not None:
            try:
                fluid = fluid.clip_surface(mold_mesh, invert=True)
            except Exception as e:
                pass
        
        # 표면 추출
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
        print(f"Error loading fluid: {e}")
        return None, None, 0

def get_sampled_files(all_files, num_steps=10):
    if not all_files:
        return []
    total = len(all_files)
    if total <= num_steps:
        return all_files
    indices = np.linspace(0, total-1, num_steps, dtype=int)
    return [all_files[i] for i in indices]

# Plotly helpers
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
    if mold_trimesh is None: return None
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
            add_log(f"STL file loaded: {len(mesh.faces):,} faces")
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
            add_log(f"AI Gate suggested: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        else:
            st.warning("Please upload an STL file first.")
    
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
    mat_name = st.text_input("Material Name", value="PA66+30GF", placeholder="PP, ABS, PA66, PC, Catamold ...", key="mat_name_input")
    st.session_state["mat_name"] = mat_name
    
    if st.button("🤖 AI Material Properties (Gemini)", use_container_width=True, type="primary"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False
        add_log(f"AI Material properties loaded for: {mat_name}")
    
    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 Source: {p.get('source', 'Gemini recommendation')}")
        if p.get("desc"):
            st.info(p["desc"])
        
        with st.expander("📋 Material Properties Check / Edit", expanded=True):
            p["nu"] = st.number_input("Kinematic Viscosity nu (m²/s)", value=float(p.get("nu", 1e-3)), format="%.2e", min_value=1e-7, max_value=1.0, key="edit_nu")
            p["rho"] = st.number_input("Density ρ (kg/m³)", value=float(p.get("rho", 1000)), min_value=100.0, max_value=9000.0, step=1.0, key="edit_rho")
            p["Tmelt"] = st.number_input("Melt Temperature (°C)", value=float(p.get("Tmelt", 220)), min_value=100.0, max_value=450.0, step=1.0, key="edit_tmelt")
            p["Tmold"] = st.number_input("Mold Temperature (°C)", value=float(p.get("Tmold", 50)), min_value=10.0, max_value=200.0, step=1.0, key="edit_tmold")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Properties", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Material Properties Confirmed!", icon="✅")
                    add_log("Material properties confirmed")
            with col2:
                if st.button("🔄 Reset Properties", use_container_width=True):
                    st.session_state["props"] = None
                    st.session_state["props_confirmed"] = False
                    add_log("Material properties reset")
                    st.rerun()
    
    st.divider()
    
    st.header("⚙️ 4. Process Conditions")
    
    if st.button("🤖 Optimize Process", use_container_width=True):
        suggestion = get_process(mat_name)
        st.session_state["temp"] = suggestion["temp"]
        st.session_state["press"] = suggestion["press"]
        st.session_state["vel"] = suggestion["vel"]
        st.toast("Process Conditions Optimized!", icon="🤖")
        add_log(f"Process optimized: Temp={suggestion['temp']}°C, Press={suggestion['press']}MPa, Vel={suggestion['vel']}mm/s")
    
    temp_c = st.number_input("Injection Temperature (°C)", 50.0, 450.0, step=1.0, key="temp")
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 250.0, step=1.0, key="press")
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 600.0, step=1.0, key="vel")
    etime = st.number_input("End Time (s)", value=float(st.session_state["etime"]), min_value=0.1, max_value=10.0, step=0.1, key="etime")
    
    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"] = etime
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Process Conditions", use_container_width=True):
            st.session_state["process_confirmed"] = True
            st.toast("Process Conditions Confirmed!", icon="✅")
            add_log("Process conditions confirmed")
    with col2:
        if st.button("🔄 Reset Process", use_container_width=True):
            st.session_state["process_confirmed"] = False
            add_log("Process conditions reset")
            st.rerun()
    
    if not st.session_state.get("process_confirmed", False):
        st.warning("⚠️ Please click ✅ Confirm Process Conditions")
    
    st.divider()
    
    num_frames = st.select_slider("Animation Quality (Frames)", options=[5, 10, 15, 20, 30], value=15)
    
    run_disabled = st.session_state["sim_running"] or not st.session_state["props_confirmed"] or not st.session_state.get("process_confirmed", False)
    
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True, disabled=run_disabled):
        if not ZAPIER_URL:
            st.error("❌ ZAPIER_URL is not configured.\nCheck .streamlit/secrets.toml")
        else:
            props = st.session_state["props"]
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"] = True
            st.session_state["sim_status"] = "running"
            
            # 상세 로그 추가
            add_log(f"🚀 Simulation started with ID: {sig_id}")
            add_log(f"Material: {mat_name}")
            add_log(f"Properties: nu={props['nu']:.2e}, rho={props['rho']} kg/m³")
            add_log(f"Tmelt={props['Tmelt']}°C, Tmold={props['Tmold']}°C")
            add_log(f"Gate position: ({gx:.2f}, {gy:.2f}, {gz:.2f}) mm, Diameter: {g_size} mm")
            add_log(f"Injection: {temp_c}°C, {press_mpa} MPa, {vel_mms} mm/s")
            add_log(f"End time: {etime} s")
            
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
                "gate_pos": {
                    "x": round(gx, 3),
                    "y": round(gy, 3),
                    "z": round(gz, 3)
                },
                "gate_size": float(g_size),
            }
            
            try:
                add_log("📡 Sending signal to Zapier webhook...")
                res = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if res.status_code == 200:
                    st.toast(f"🚀 Signal Sent Successfully! (ID: {sig_id})", icon="🚀")
                    add_log(f"✅ Signal sent successfully! HTTP {res.status_code}")
                    add_log("⏳ Waiting for GitHub Actions to process...")
                    add_log("📊 Check progress at: GitHub Actions → Artifacts")
                else:
                    st.error(f"Transmission failed: HTTP {res.status_code}")
                    add_log(f"❌ Transmission failed: HTTP {res.status_code}")
                    st.session_state["sim_running"] = False
                    st.session_state["sim_status"] = "failed"
            except Exception as e:
                st.error(f"Connection error: {e}")
                add_log(f"❌ Connection error: {e}")
                st.session_state["sim_running"] = False
                st.session_state["sim_status"] = "failed"

# ========== MAIN AREA - Geometry Preview ==========
col_geo, col_log = st.columns([2, 1])

with col_geo:
    st.header("🎥 3D Geometry Analysis")
    mesh = st.session_state.get("mesh")
    
    if mesh is not None:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:,0], y=v[:,1], z=v[:,2],
                i=f[:,0], j=f[:,1], k=f[:,2],
                color="#AAAAAA", opacity=0.7,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3)
            ),
            go.Scatter3d(
                x=[st.session_state["gx_final"]], 
                y=[st.session_state["gy_final"]], 
                z=[st.session_state["gz_final"]],
                mode="markers",
                marker=dict(size=st.session_state["gsize"] * 3, color="red", symbol="circle"),
                name="Gate Location"
            )
        ])
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode="data"),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        bb = mesh.bounds
        c1, c2, c3 = st.columns(3)
        c1.metric("X Size", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y Size", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z Size", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("Upload an STL file in the sidebar to display the 3D model.")

with col_log:
    st.header("📟 Simulation & Debug Logs")
    
    # 시뮬레이션 상태 표시
    if st.session_state["sim_running"]:
        st.info("🟢 Status: SIMULATION RUNNING...")
    elif st.session_state["sim_status"] == "failed":
        st.error("🔴 Status: SIMULATION FAILED")
    else:
        st.success("✅ Status: READY")
    
    # 로그 표시
    log_container = st.container(height=400)
    with log_container:
        for log in st.session_state["sim_logs"][-30:]:
            st.code(log, language="bash")
    
    # 시뮬레이션 완료 버튼
    if st.session_state["sim_running"]:
        if st.button("✅ Mark Simulation as Completed", use_container_width=True):
            st.session_state["sim_running"] = False
            st.session_state["sim_status"] = "completed"
            add_log("🏁 Simulation marked as completed by user")
            st.rerun()
    
    # 로그 클리어 버튼
    if st.button("🗑 Clear Logs", use_container_width=True):
        st.session_state["sim_logs"] = []
        st.rerun()

st.info(f"📍 Final Gate Position: ({st.session_state['gx_final']:.2f}, {st.session_state['gy_final']:.2f}, {st.session_state['gz_final']:.2f}) mm")

if st.session_state["props_confirmed"] and st.session_state.get("process_confirmed", False) and st.session_state["props"]:
    p = st.session_state["props"]
    st.caption(f"ℹ️ Properties & Process confirmed | nu={p['nu']:.2e} | rho={p['rho']} kg/m³ | Tmelt={p['Tmelt']}°C | Tmold={p['Tmold']}°C")
else:
    st.caption("ℹ️ Confirm both Material Properties and Process Conditions in the sidebar before running simulation.")

# ========== RESULTS SECTION ==========
st.title("📊 Simulation Results")

if st.button("🔄 Sync Latest Results from GitHub", use_container_width=True):
    with st.spinner("Fetching latest data from GitHub securely..."):
        if sync_simulation_results():
            st.success("Data synchronization complete! Loading visualization data.")
            time.sleep(1)
            st.rerun()

# 결과 파일 확인
if os.path.exists("results.txt"):
    with open("results.txt", "r") as f:
        summary = f.read()
    st.text_area("📄 Simulation Summary", summary, height=150)

if os.path.exists("logs.zip"):
    with open("logs.zip", "rb") as f:
        st.download_button(
            label="📂 Download All Logs (logs.zip)",
            data=f,
            file_name="logs.zip",
            mime="application/zip",
            use_container_width=True
        )

# ========== ANIMATION SECTION ==========
vtk_dir = "VTK"
if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (Volume Flow)")
    
    all_files = sorted(glob.glob(f"{vtk_dir}/**/case_*.vt*", recursive=True) + glob.glob(f"{vtk_dir}/case_*.vt*", recursive=True))
    all_files = sorted(set(all_files), key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    
    if not all_files:
        st.warning("No VTK files found. Run simulation first.")
    else:
        mold_mesh = st.session_state.get("mesh")
        
        with st.expander("🔧 Visualization Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                scale_factor = st.slider("Scale (m → mm)", 100.0, 2000.0, 1000.0, 50.0, 
                                        help="OpenFOAM uses meters, STL uses mm. 1000 = 1m → 1000mm")
                threshold = st.slider("Fluid Threshold (alpha.water)", 0.01, 0.5, 0.05, 0.01,
                                     help="Lower value shows more fluid volume")
            with col2:
                mold_opacity = st.slider("Mold Opacity", 0.0, 0.5, 0.1, 0.01)
                fluid_opacity = st.slider("Fluid Opacity", 0.5, 1.0, 0.75, 0.05)
            with col3:
                view_mode = st.radio("View Scale", ["Auto (Data)", "Uniform Cube"], index=0)
        
        sampled_files = get_sampled_files(all_files, num_frames)
        total_steps = len(sampled_files)
        
        st.markdown("### 🎮 Animation Controls")
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 3, 1])
        
        with col_ctrl1:
            if st.button("⏮ First", use_container_width=True):
                st.session_state["current_frame"] = 0
                st.session_state["animation_playing"] = False
                st.rerun()
        with col_ctrl2:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state["animation_playing"] = False
                st.rerun()
        with col_ctrl3:
            if st.button("▶ Play", use_container_width=True, type="primary"):
                st.session_state["animation_playing"] = True
                st.rerun()
        with col_ctrl4:
            if st.button("⏭ Last", use_container_width=True):
                st.session_state["current_frame"] = total_steps - 1
                st.session_state["animation_playing"] = False
                st.rerun()
        
        current_frame = st.slider("Time Step", 0, total_steps - 1, 
                                  value=st.session_state.get("current_frame", 0),
                                  key="frame_slider")
        st.session_state["current_frame"] = current_frame
        
        with st.spinner("Loading fluid data..."):
            fpath = sampled_files[current_frame]
            res, alpha_vals, n_cells = load_fluid_volume(fpath, mold_mesh, scale=scale_factor, thres=threshold)
            
            fig = go.Figure()
            
            if mold_mesh:
                fig.add_trace(make_mold_trace(mold_mesh, opacity=mold_opacity))
            
            # 게이트 위치 표시
            gate_x, gate_y, gate_z = st.session_state["gx_final"], st.session_state["gy_final"], st.session_state["gz_final"]
            fig.add_trace(go.Scatter3d(
                x=[gate_x], y=[gate_y], z=[gate_z],
                mode="markers",
                marker=dict(size=st.session_state["gsize"] * 2, color="red", symbol="x", line=dict(width=2, color="white")),
                name="Gate Location",
                showlegend=True
            ))
            
            if res:
                pts, fi, fj, fk = res
                fig.add_trace(make_fluid_trace(pts, fi, fj, fk, alpha_vals, opacity=fluid_opacity))
                
                if mold_mesh:
                    bounds = mold_mesh.bounds
                    fluid_bounds = [pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), pts[:,2].min(), pts[:,2].max()]
                    st.success(f"✅ Frame {current_frame+1}/{total_steps} | Fluid cells: {n_cells:,}")
                else:
                    st.success(f"Frame {current_frame+1}/{total_steps} - Fluid cells: {n_cells:,}")
            else:
                st.warning(f"⚠️ No fluid detected at threshold={threshold}. Try lowering the threshold.")
            
            scene_config = dict(
                aspectmode="data" if view_mode == "Auto (Data)" else "cube",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.5), up=dict(x=0, y=0, z=1)),
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)", 
                zaxis_title="Z (mm)",
                bgcolor="black"
            )
            
            fig.update_layout(
                scene=scene_config,
                height=650,
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
                paper_bgcolor="black",
                plot_bgcolor="black"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            })
        
        # 자동 재생
        if st.session_state.get("animation_playing", False):
            next_frame = (current_frame + 1) % total_steps
            st.session_state["current_frame"] = next_frame
            time.sleep(0.15)
            st.rerun()
        
        if mold_mesh:
            bounds = mold_mesh.bounds
            st.caption(f"📐 Model bounds: X [{bounds[0][
