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
from plotly.subplots import make_subplots
import trimesh
import meshio

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# Session state init
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0); _init("gy", 0.0); _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp", 230); _init("press", 70.0); _init("vel", 80.0)
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
_init("animation_playing", False)
_init("current_frame", 0)

# Material Database (same as original)
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
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    return {"nu": 1e-3, "rho": 1000, "Tmelt": 220, "Tmold": 50, "press_mpa": 70, "vel_mms": 80, "material": material, "source": "Gemini recommendation", "desc": f"{material} — default"}

def get_process(material: str) -> dict:
    props = get_props(material)
    return {"temp": props.get("Tmelt", 230), "press": float(props.get("press_mpa", 70)), "vel": float(props.get("vel_mms", 80))}

# GitHub sync
def sync_simulation_results():
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = "workshopcompany"
    REPO_NAME = "OpenFOAM-Injection-Automation"
    ARTIFACT_NAME = "simulation-results"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"GitHub API failed: {response.status_code}")
        return False
    artifacts = response.json().get("artifacts", [])
    target = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
    if not target:
        st.warning("No simulation results yet.")
        return False
    file_res = requests.get(target["archive_download_url"], headers=headers)
    if file_res.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(file_res.content)) as z:
            z.extractall(".")
        return True
    return False

# ========== 3D 유체 로드 함수 (개선됨) ==========
def load_fluid_3d_clipped(fpath, mold_mesh=None, scale=1.0, thres=0.5):
    """VTK에서 3D 유체 표면 생성 및 클리핑"""
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        
        f_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if f_name not in mesh.array_names:
            return None, None, 0
        
        # 등치면 생성
        fluid_3d = mesh.contour([thres], scalars=f_name)
        if fluid_3d.n_points == 0:
            return None, None, 0
        
        # 스케일 조정 (기본값 1, 사용자가 조절 가능)
        fluid_3d.points *= scale
        
        # 금형 클리핑
        if mold_mesh is not None:
            try:
                fluid_3d = fluid_3d.clip_surface(mold_mesh, invert=True)
            except:
                pass
        
        surf = fluid_3d.triangulate()
        pts = surf.points
        if len(pts) == 0:
            return None, None, 0
            
        faces = surf.faces.reshape(-1, 4)[:, 1:] if surf.faces.size > 0 else np.empty((0,3), dtype=int)
        alpha = surf.point_data[f_name].tolist() if f_name in surf.point_data else [0.5]*len(pts)
        
        return (pts, faces[:,0], faces[:,1], faces[:,2]), alpha, fluid_3d.n_cells
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

# Plotly helpers
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

def _safe_json_fixed(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder).replace('</', '<\\/')

def make_fluid_trace(pts, fi, fj, fk, alpha_vals, name="Fluid", opacity=0.8):
    """유체 트레이스 생성 - 색상으로 농도 표현"""
    intensity = np.array(alpha_vals) if alpha_vals is not None else np.ones(len(pts))
    return go.Mesh3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        i=fi, j=fj, k=fk,
        intensity=intensity,
        colorscale="Viridis",
        opacity=opacity,
        name=name,
        showscale=True,
        colorbar=dict(title="Fill Ratio", thickness=20, len=0.5)
    )

def make_mold_trace(mold_trimesh, opacity=0.15, color="lightgray"):
    """금형 트레이스 - 반투명"""
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
            st.success(f"✅ Loaded: {len(mesh.faces):,} faces")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.divider()
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion"):
        mesh = st.session_state.get("mesh")
        if mesh:
            center = mesh.centroid
            snap, _, _ = trimesh.proximity.closest_point(mesh, [center])
            pos = snap[0]
            st.session_state["gx"] = float(pos[0])
            st.session_state["gy"] = float(pos[1])
            st.session_state["gz"] = float(pos[2])
            st.toast("Gate position suggested!", icon="🪄")
    
    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, 2.0, key="gsize")
    vx = st.number_input("Gate X", value=st.session_state["gx"], key="gx")
    vy = st.number_input("Gate Y", value=st.session_state["gy"], key="gy")
    vz = st.number_input("Gate Z", value=st.session_state["gz"], key="gz")
    
    mesh = st.session_state.get("mesh")
    if mesh:
        snap, _, _ = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])
        gx, gy, gz = snap[0]
    else:
        gx, gy, gz = vx, vy, vz
    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz
    
    st.divider()
    st.header("🧪 3. Material")
    mat_name = st.text_input("Material", value="PA66+30GF", key="mat_name")
    if st.button("🤖 AI Material Properties"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False
    
    if st.session_state["props"]:
        p = st.session_state["props"]
        with st.expander("📋 Edit Properties", expanded=True):
            p["nu"] = st.number_input("Viscosity (m²/s)", value=float(p["nu"]), format="%.2e")
            p["rho"] = st.number_input("Density (kg/m³)", value=float(p["rho"]))
            p["Tmelt"] = st.number_input("Melt Temp (°C)", value=int(p["Tmelt"]))
            if st.button("✅ Confirm Properties"):
                st.session_state["props_confirmed"] = True
                st.toast("Properties confirmed!")
    
    st.divider()
    st.header("⚙️ 4. Process")
    if st.button("🤖 Optimize Process"):
        opt = get_process(mat_name)
        st.session_state["temp"] = opt["temp"]
        st.session_state["press"] = opt["press"]
        st.session_state["vel"] = opt["vel"]
    
    temp_c = st.number_input("Temp (°C)", 50, 450, st.session_state["temp"], key="temp")
    press_mpa = st.number_input("Pressure (MPa)", 10, 250, st.session_state["press"], key="press")
    vel_mms = st.number_input("Velocity (mm/s)", 1, 600, st.session_state["vel"], key="vel")
    
    if st.button("✅ Confirm Process"):
        st.session_state["process_confirmed"] = True
        st.toast("Process confirmed!")
    
    st.divider()
    
    # 프레임 수 선택
    num_frames = st.select_slider("Animation Frames", options=[5, 10, 15, 20, 30], value=15)
    
    if st.button("🚀 Run Simulation", disabled=not (st.session_state["props_confirmed"] and st.session_state.get("process_confirmed"))):
        if ZAPIER_URL:
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"] = True
            # Send payload...
            st.toast(f"Simulation started! ID: {sig_id}")
        else:
            st.error("ZAPIER_URL not configured")

# ========== MAIN AREA - Geometry Preview ==========
col_geo, col_log = st.columns([2, 1])
with col_geo:
    st.header("🎥 3D Model & Gate")
    mesh = st.session_state.get("mesh")
    if mesh:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2], color="gray", opacity=0.7),
            go.Scatter3d(x=[st.session_state["gx_final"]], y=[st.session_state["gy_final"]], z=[st.session_state["gz_final"]],
                        mode="markers", marker=dict(size=10, color="red"), name="Gate")
        ])
        fig.update_layout(scene=dict(aspectmode="data"), height=500, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload STL file to see 3D model")

with col_log:
    st.header("📟 Status")
    if st.session_state["sim_running"]:
        st.code(f"Running simulation...\nID: {st.session_state['last_signal_id']}", language="bash")
        if st.button("✅ Mark Complete"):
            st.session_state["sim_running"] = False
            st.rerun()
    else:
        st.info("Ready. Configure and run simulation.")

# ========== RESULTS SECTION ==========
st.title("📊 Simulation Results")

if st.button("🔄 Sync from GitHub"):
    with st.spinner("Syncing..."):
        if sync_simulation_results():
            st.success("Synced!")
            st.rerun()

if os.path.exists("results.txt"):
    with open("results.txt") as f:
        st.text_area("Summary", f.read(), height=150)

# ========== ANIMATION SECTION ==========
vtk_dir = "VTK"
if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation")
    
    # VTK 파일 찾기
    all_files = sorted(glob.glob(f"{vtk_dir}/**/case_*.vt*", recursive=True) + glob.glob(f"{vtk_dir}/case_*.vt*", recursive=True))
    all_files = sorted(set(all_files), key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    
    if not all_files:
        st.warning("No VTK files found")
    else:
        mold_mesh = st.session_state.get("mesh")
        
        # 시각화 설정
        with st.expander("🔧 Visualization Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                scale_factor = st.slider("Model Scale", 0.1, 10.0, 1.0, help="Adjust if fluid is too small")
                contour_thresh = st.slider("Fill Threshold", 0.01, 0.99, 0.3, help="Lower = more fluid visible")
            with col2:
                mold_opacity = st.slider("Mold Opacity", 0.0, 0.5, 0.1)
                fluid_opacity = st.slider("Fluid Opacity", 0.5, 1.0, 0.8)
            with col3:
                view_scale = st.radio("View Scale", ["Auto", "Uniform"], index=0)
        
        # 샘플링된 파일
        sampled_files = get_sampled_files(all_files, num_frames)
        total_steps = len(sampled_files)
        
        # ===== 프레임 컨트롤 UI =====
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
        
        # 프레임 슬라이더
        current_frame = st.slider(
            "Frame", 0, total_steps - 1, 
            value=st.session_state.get("current_frame", 0),
            key="frame_slider",
            on_change=lambda: st.session_state.update({"animation_playing": False, "current_frame": st.session_state.frame_slider})
        )
        
        # 현재 프레임 로드 및 표시
        with st.spinner("Loading frame..."):
            fpath = sampled_files[current_frame]
            res, alpha_vals, n_cells = load_fluid_3d_clipped(fpath, mold_mesh, scale=scale_factor, thres=contour_thresh)
            
            # 3D 플롯 생성
            fig = go.Figure()
            
            # 금형 추가
            if mold_mesh:
                fig.add_trace(make_mold_trace(mold_mesh, opacity=mold_opacity))
            
            # 유체 추가 (있을 경우)
            if res:
                pts, fi, fj, fk = res
                fig.add_trace(make_fluid_trace(pts, fi, fj, fk, alpha_vals, opacity=fluid_opacity))
                st.success(f"Frame {current_frame+1}/{total_steps} - Fluid cells: {n_cells:,}")
            else:
                st.warning(f"Frame {current_frame+1}: No fluid detected (threshold may be too high)")
            
            # 레이아웃 설정 - 확대/축소/회전 가능하도록!
            scene_config = dict(
                aspectmode="data" if view_scale == "Auto" else "cube",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)", 
                zaxis_title="Z (mm)"
            )
            
            fig.update_layout(
                scene=scene_config,
                height=650,
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            })
        
        # ===== 자동 재생 로직 =====
        if st.session_state.get("animation_playing", False):
            next_frame = (current_frame + 1) % total_steps
            st.session_state["current_frame"] = next_frame
            st.rerun()
        
        # ===== Bounding Box 정보 =====
        if mold_mesh:
            bounds = mold_mesh.bounds
            st.caption(f"📐 Model bounds: X [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}] | Y [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}] | Z [{bounds[0][2]:.1f}, {bounds[1][2]:.1f}] mm")
            
            # 스케일 추천
            max_dim = max(bounds[1][0]-bounds[0][0], bounds[1][1]-bounds[0][1], bounds[1][2]-bounds[0][2])
            recommended_scale = 1000.0 / max_dim if max_dim > 0 else 1.0
            st.info(f"💡 Tip: If fluid is too small, try Scale ≈ {recommended_scale:.1f}")

else:
    st.error("VTK directory not found. Run simulation and sync results first.")
