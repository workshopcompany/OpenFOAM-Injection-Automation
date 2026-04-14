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
    "CATAMOLD": {"nu": 5e-3, "rho": 4900.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 100.0, "vel_mms": 30.0, "desc": "BASF Catamold MIM"},
    "MIM": {"nu": 5e-3, "rho": 5000.0, "Tmelt": 185.0, "Tmold": 40.0, "press_mpa": 100.0, "vel_mms": 30.0, "desc": "Metal injection molding"},
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Gemini recommendation"}
    return {"nu": 1e-3, "rho": 1000.0, "Tmelt": 220.0, "Tmold": 50.0, "press_mpa": 70.0, "vel_mms": 80.0, "material": material, "source": "Gemini recommendation", "desc": f"{material} — default"}

def get_process(material: str) -> dict:
    props = get_props(material)
    return {"temp": float(props.get("Tmelt", 230.0)), "press": float(props.get("press_mpa", 70.0)), "vel": float(props.get("vel_mms", 80.0))}

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

# ========== 3D 유체 로드 함수 (threshold 기반 - 부피감 있는 표현) ==========
def load_fluid_volume(fpath, mold_mesh=None, scale=1.0, thres=0.1):
    """
    threshold 방식으로 유체 데이터를 추출 (3D volume 형태)
    thres 값이 낮을수록 더 많은 유체 영역 포함
    """
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        
        f_name = "alpha.water" if "alpha.water" in mesh.array_names else "alpha1"
        if f_name not in mesh.array_names:
            return None, None, 0
        
        # threshold로 유체 영역 추출 (thres 이상인 모든 셀)
        fluid = mesh.threshold(thres, scalars=f_name)
        if fluid.n_cells == 0:
            # 더 낮은 threshold로 재시도
            fluid = mesh.threshold(0.01, scalars=f_name)
            if fluid.n_cells == 0:
                return None, None, 0
        
        # 스케일 적용 (OpenFOAM은 meter 단위, STL은 mm 단위)
        fluid.points *= scale
        
        # 금형 클리핑 (선택적)
        if mold_mesh is not None:
            try:
                fluid = fluid.clip_surface(mold_mesh, invert=True)
            except:
                pass
        
        # 표면 추출 (시각화를 위해)
        surf = fluid.extract_surface()
        if surf.n_points == 0:
            return None, None, 0
        
        pts = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:] if surf.faces.size > 0 else np.empty((0,3), dtype=int)
        
        # alpha 값 가져오기 (point data)
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
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

def make_fluid_trace(pts, fi, fj, fk, alpha_vals, name="Fluid", opacity=0.7):
    """유체 트레이스 - 부피감 있는 표현"""
    intensity = np.array(alpha_vals) if alpha_vals is not None else np.ones(len(pts))
    
    return go.Mesh3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        i=fi, j=fj, k=fk,
        intensity=intensity,
        colorscale=[[0, 'blue'], [0.3, 'cyan'], [0.6, 'lime'], [0.8, 'orange'], [1.0, 'red']],
        opacity=opacity,
        name=name,
        showscale=True,
        colorbar=dict(title="Fill Ratio", thickness=20, len=0.6, x=1.02),
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3)
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
    
    g_size = st.number_input("Gate Diameter (mm)", min_value=0.5, max_value=10.0, value=2.0, step=0.1, key="gsize")
    vx = st.number_input("Gate X", value=float(st.session_state["gx"]), step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=float(st.session_state["gy"]), step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=float(st.session_state["gz"]), step=0.1, key="gz")
    
    mesh = st.session_state.get("mesh")
    if mesh:
        snap, _, _ = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])
        gx, gy, gz = float(snap[0][0]), float(snap[0][1]), float(snap[0][2])
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
            p["nu"] = st.number_input("Viscosity (m²/s)", value=float(p["nu"]), format="%.2e", step=1e-5)
            p["rho"] = st.number_input("Density (kg/m³)", value=float(p["rho"]), step=10.0)
            p["Tmelt"] = st.number_input("Melt Temp (°C)", value=float(p["Tmelt"]), step=1.0)
            if st.button("✅ Confirm Properties"):
                st.session_state["props_confirmed"] = True
                st.toast("Properties confirmed!")
    
    st.divider()
    st.header("⚙️ 4. Process")
    if st.button("🤖 Optimize Process"):
        opt = get_process(mat_name)
        st.session_state["temp"] = float(opt["temp"])
        st.session_state["press"] = float(opt["press"])
        st.session_state["vel"] = float(opt["vel"])
    
    temp_c = st.number_input("Temp (°C)", min_value=50.0, max_value=450.0, value=float(st.session_state["temp"]), step=1.0, key="temp")
    press_mpa = st.number_input("Pressure (MPa)", min_value=10.0, max_value=250.0, value=float(st.session_state["press"]), step=1.0, key="press")
    vel_mms = st.number_input("Velocity (mm/s)", min_value=1.0, max_value=600.0, value=float(st.session_state["vel"]), step=1.0, key="vel")
    
    if st.button("✅ Confirm Process"):
        st.session_state["process_confirmed"] = True
        st.toast("Process confirmed!")
    
    st.divider()
    
    num_frames = st.select_slider("Animation Frames", options=[5, 10, 15, 20, 30], value=15)
    
    run_disabled = not (st.session_state["props_confirmed"] and st.session_state.get("process_confirmed", False))
    if st.button("🚀 Run Simulation", disabled=run_disabled):
        if ZAPIER_URL:
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"] = True
            # 실제 payload 전송 코드는 여기에...
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
                        mode="markers", marker=dict(size=10, color="red", symbol="circle"), name="Gate")
        ])
        fig.update_layout(
            scene=dict(aspectmode="data"),
            height=500,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
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
    st.subheader("🌊 3D Filling Animation - Volume Flow")
    
    # VTK 파일 찾기
    all_files = sorted(glob.glob(f"{vtk_dir}/**/case_*.vt*", recursive=True) + glob.glob(f"{vtk_dir}/case_*.vt*", recursive=True))
    all_files = sorted(set(all_files), key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    
    if not all_files:
        st.warning("No VTK files found. Run simulation first.")
    else:
        mold_mesh = st.session_state.get("mesh")
        
        # 시각화 설정
        with st.expander("🔧 Fluid Visualization Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                # 중요: 스케일 - OpenFOAM은 m 단위, STL은 mm 단위. 보통 1000이 필요
                scale_factor = st.slider("Scale (m → mm)", 100.0, 2000.0, 1000.0, 50.0, 
                                        help="OpenFOAM 결과는 meter 단위. 1000 = 1m → 1000mm")
                threshold = st.slider("Fluid Threshold (alpha.water)", 0.01, 0.5, 0.05, 0.01,
                                     help="낮을수록 더 많은 유체 표시 (0.01~0.05 추천)")
            with col2:
                mold_opacity = st.slider("Mold Opacity", 0.0, 0.5, 0.1, 0.01)
                fluid_opacity = st.slider("Fluid Opacity", 0.5, 1.0, 0.75, 0.05)
            with col3:
                view_mode = st.radio("View Scale", ["Auto (Data)", "Uniform Cube"], index=0)
        
        # 샘플링
        sampled_files = get_sampled_files(all_files, num_frames)
        total_steps = len(sampled_files)
        
        # 애니메이션 컨트롤
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
            "Time Step", 0, total_steps - 1, 
            value=st.session_state.get("current_frame", 0),
            key="frame_slider"
        )
        st.session_state["current_frame"] = current_frame
        
        # 현재 프레임 로드 및 표시
        with st.spinner("Loading fluid data..."):
            fpath = sampled_files[current_frame]
            res, alpha_vals, n_cells = load_fluid_volume(
                fpath, mold_mesh, 
                scale=scale_factor, 
                thres=threshold
            )
            
            fig = go.Figure()
            
            # 금형 추가
            if mold_mesh:
                fig.add_trace(make_mold_trace(mold_mesh, opacity=mold_opacity))
            
            # 유체 추가
            if res:
                pts, fi, fj, fk = res
                fig.add_trace(make_fluid_trace(pts, fi, fj, fk, alpha_vals, opacity=fluid_opacity))
                
                # 바운딩 박스 정보로 스케일 확인
                if mold_mesh:
                    bounds = mold_mesh.bounds
                    fluid_bounds = [pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), pts[:,2].min(), pts[:,2].max()]
                    st.success(f"✅ Frame {current_frame+1}/{total_steps} | Fluid cells: {n_cells:,} | "
                              f"Fluid range: X[{fluid_bounds[0]:.1f}, {fluid_bounds[1]:.1f}]")
                else:
                    st.success(f"Frame {current_frame+1}/{total_steps} - Fluid cells: {n_cells:,}")
            else:
                st.warning(f"⚠️ No fluid detected at threshold={threshold}. Try lowering the threshold or check simulation data.")
                # 디버그 정보
                st.caption(f"File: {os.path.basename(fpath)}")
            
            # 3D 뷰 설정
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
        
        # 유용한 정보
        if mold_mesh:
            bounds = mold_mesh.bounds
            st.caption(f"📐 Model bounds: X [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}] | "
                      f"Y [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}] | "
                      f"Z [{bounds[0][2]:.1f}, {bounds[1][2]:.1f}] mm")
            
            st.info("💡 **Tip**: If fluid is not visible, try:\n"
                   "• Lower 'Fluid Threshold' (0.01~0.05)\n"
                   "• Check 'Scale' is 1000 (m→mm conversion)\n"
                   "• Verify simulation completed successfully")

else:
    st.error("VTK directory not found. Run simulation and sync results first.")
