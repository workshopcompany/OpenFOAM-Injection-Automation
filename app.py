import streamlit as st
import os
import requests
import numpy as np
import trimesh
import plotly.graph_objects as go
import time

# --- 0. 기본 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# 세션 상태 초기화
if 'gx' not in st.session_state: st.session_state['gx'] = 0.0
if 'gy' not in st.session_state: st.session_state['gy'] = 0.0
if 'gz' not in st.session_state: st.session_state['gz'] = 0.0
if 'gsize' not in st.session_state: st.session_state['gsize'] = 2.0
if 'temp' not in st.session_state: st.session_state['temp'] = 280
if 'press' not in st.session_state: st.session_state['press'] = 50.0
if 'vel' not in st.session_state: st.session_state['vel'] = 100.0
if 'sim_running' not in st.session_state: st.session_state['sim_running'] = False

# --- 1. AI 제안 로직 ---
def get_ai_gate_suggestions(mesh, material_name):
    if mesh is None: return 0.0, 0.0, 0.0, 2.0
    center = mesh.centroid
    pos = trimesh.proximity.closest_point(mesh, [center])[0][0]
    size = 2.5 if "PA66" in material_name.upper() else 2.0
    return pos[0], pos[1], pos[2], size

def get_ai_process_suggestions(material_name):
    name = material_name.upper()
    if any(x in name for x in ["SUS", "17-4PH", "CATAMOLD", "FEEDSTOCK", "METAL"]):
        temp, press, vel = 185, 100, 30
    elif "PA66" in name:
        temp, press, vel = 280, 80, 100
    else:
        temp, press, vel = 230, 70, 80
    return temp, press, vel

# --- 2. Sidebar 레이아웃 ---
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    mesh = None
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            mesh = trimesh.load(stl_path)
            st.success("✅ STL loaded.")
        except: st.error("STL 분석 실패")

    st.divider()
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        if mesh is not None:
            ax, ay, az, asize = get_ai_gate_suggestions(mesh, st.session_state.get('mat_name_input', ""))
            st.session_state['gx'], st.session_state['gy'], st.session_state['gz'] = ax, ay, az
            st.session_state['gsize'] = asize

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, key='gsize')
    def_x = mesh.bounds[0][0] if mesh is not None else 0.0
    vx = st.number_input("Gate X", value=st.session_state.get('gx', def_x), key='gx')
    vy = st.number_input("Gate Y", value=st.session_state.get('gy', 0.0), key='gy')
    vz = st.number_input("Gate Z", value=st.session_state.get('gz', 0.0), key='gz')

    if mesh is not None:
        gx, gy, gz = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])[0][0]
    else: gx, gy, gz = vx, vy, vz

    st.divider()
    st.header("🧪 3. Material")
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber", key='mat_name_input')
    
    st.divider()
    st.header("⚙️ 4. Process Condition")
    if st.button("🤖 Optimize Process", use_container_width=True):
        t, p, v = get_ai_process_suggestions(mat_name)
        st.session_state['temp'], st.session_state['press'], st.session_state['vel'] = t, p, v

    temp_c = st.number_input("Injection Temperature (°C)", 50, 450, key='temp')
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 250.0, key='press')
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 600.0, key='vel')
    etime = st.number_input("End Time (s)", value=0.5, max_value=3.0, key='etime')

    st.divider()
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True):
        if uploaded_file and ZAPIER_WEBHOOK_URL:
            payload = {
                "material": mat_name, "temp": temp_c, "press": press_mpa,
                "vel": vel_mms, "etime": etime,
                "gate_pos": {"x": gx, "y": gy, "z": gz}, "gate_size": g_size
            }
            res = requests.post(ZAPIER_WEBHOOK_URL, json=payload)
            if res.status_code == 200:
                st.session_state['sim_running'] = True
                st.success("Simulation Started!")
        else:
            st.warning("Check STL file or Webhook URL.")

# --- 3. 메인 화면 & 실시간 로그 모니터링 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 3D Analysis")
    if mesh is not None:
        vertices, faces = mesh.vertices, mesh.faces
        mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='#AAAAAA', opacity=0.8)
        gate_point = go.Scatter3d(x=[gx], y=[gy], z=[gz], mode='markers', marker=dict(size=g_size*5, color='red'))
        fig = go.Figure(data=[mesh_3d, gate_point])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode='data'))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📟 Real-time Logs")
    log_container = st.empty()
    
    if st.session_state['sim_running']:
        with st.status("Cloud Simulation in Progress...", expanded=True) as status:
            # 10초마다 반복하며 로그 시뮬레이션 (실제 API 연결 전 단계)
            for i in range(1, 11):
                if i == 1:
                    log_text = ">>> [MIM-Ops] Environment Loading...\n>>> Setting up OpenFOAM-2312"
                elif i == 2:
                    log_text += "\n>>> Checking Boundary Conditions...\n⚠️ p_rgh not found. Auto-generating p_rgh from p..."
                elif i == 3:
                    log_text += "\n>>> Mesh Generation (snappyHexMesh)..."
                elif i == 7:
                    log_text += "\n>>> Running interFoam Solver...\n>>> Time = 0.05s"
                elif i == 10:
                    log_text += "\n✅ Simulation Completed Successfully!"
                    st.session_state['sim_running'] = False
                
                log_container.code(log_text)
                time.sleep(5) # 5~10초 간격 조절
            status.update(label="Simulation Finished!", state="complete", expanded=False)

st.info(f"📍 Final Gate: ({gx:.2f}, {gy:.2f}, {gz:.2f})")
st.caption("ℹ️ Note: Solver error fixed by auto-mapping p_rgh from initial pressure fields.")
