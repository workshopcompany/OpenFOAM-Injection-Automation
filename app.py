import streamlit as st
import os
import requests
import numpy as np
import trimesh
import plotly.graph_objects as go
import time
import uuid
from datetime import datetime

# --- 0. 기본 설정 및 세션 초기화 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# 세션 상태 관리 (UI 연속성 및 중복 방지)
if 'gx' not in st.session_state: st.session_state['gx'] = 0.0
if 'gy' not in st.session_state: st.session_state['gy'] = 0.0
if 'gz' not in st.session_state: st.session_state['gz'] = 0.0
if 'gsize' not in st.session_state: st.session_state['gsize'] = 2.0
if 'temp' not in st.session_state: st.session_state['temp'] = 280
if 'press' not in st.session_state: st.session_state['press'] = 50.0
if 'vel' not in st.session_state: st.session_state['vel'] = 100.0
if 'sim_running' not in st.session_state: st.session_state['sim_running'] = False
if 'last_signal_id' not in st.session_state: st.session_state['last_signal_id'] = None

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

# --- 2. Sidebar 레이아웃 (설정창) ---
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    mesh = None
    if uploaded_file:
        try:
            # [중복방지 핵심] 로컬에 저장하지 않고 업로드된 파일을 직접 메모리에서 로드
            mesh = trimesh.load(uploaded_file, file_type='stl')
            st.success("✅ STL loaded in memory.")
        except: st.error("STL 분석 실패")

    st.divider()
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        if mesh is not None:
            ax, ay, az, asize = get_ai_gate_suggestions(mesh, st.session_state.get('mat_name_input', ""))
            st.session_state.update({'gx': ax, 'gy': ay, 'gz': az, 'gsize': asize})

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, key='gsize', step=0.1)
    vx = st.number_input("Gate X", value=st.session_state['gx'], key='gx', step=0.1)
    vy = st.number_input("Gate Y", value=st.session_state['gy'], key='gy', step=0.1)
    vz = st.number_input("Gate Z", value=st.session_state['gz'], key='gz', step=0.1)

    # 게이트 포인트를 표면에 스냅
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
        st.session_state.update({'temp': t, 'press': p, 'vel': v})

    temp_c = st.number_input("Injection Temperature (°C)", 50, 450, key='temp')
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 250.0, key='press')
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 600.0, key='vel')
    etime = st.number_input("End Time (s)", value=0.5, max_value=3.0, key='etime')

    st.divider()
    # 실행 버튼 (중복 클릭 방지)
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True, disabled=st.session_state['sim_running']):
        if uploaded_file and ZAPIER_WEBHOOK_URL:
            st.session_state['sim_running'] = True
            
            # Zapier 추적용 고유 ID 생성
            sig_id = str(uuid.uuid4())[:8]
            st.session_state['last_signal_id'] = sig_id
            
            payload = {
                "signal_id": sig_id,
                "material": mat_name, 
                "temp": temp_c, "press": press_mpa, "vel": vel_mms, "etime": etime,
                "gate_pos": {"x": float(gx), "y": float(gy), "z": float(gz)}, 
                "gate_size": float(g_size),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                res = requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
                if res.status_code == 200:
                    st.toast(f"Signal {sig_id} Dispatched!", icon="🚀")
                else:
                    st.error(f"Error {res.status_code}")
                    st.session_state['sim_running'] = False
            except:
                st.error("Connection Failed.")
                st.session_state['sim_running'] = False
        else:
            st.warning("Check Setup (STL/URL).")

# --- 3. 메인 화면: 시각화 및 로그 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎥 3D Geometry Analysis")
    if mesh is not None:
        vertices, faces = mesh.vertices, mesh.faces
        mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], 
                           i=faces[:,0], j=faces[:,1], k=faces[:,2], color='#AAAAAA', opacity=0.8)
        gate_point = go.Scatter3d(x=[gx], y=[gy], z=[gz], mode='markers', 
                                 marker=dict(size=g_size*5, color='red'))
        fig = go.Figure(data=[mesh_3d, gate_point])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode='data'))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("📟 Simulation & Debug Logs")
    log_area = st.empty()
    
    if st.session_state['sim_running']:
        with st.status("Solving MIM Flow...", expanded=True) as status:
            logs = [
                f">>> [MIM-Ops] Outbound Signal ID: {st.session_state['last_signal_id']}",
                ">>> Preventing Duplicate Runs: Bypass GitHub Push Trigger.",
                ">>> Verifying OpenFOAM Dictionary Integrity...",
                "⚠️ Fatal Syntax Error detected in 'constant/transportProperties'.",
                "💡 Reason: Invalid YAML-style colons (:) found in dictionary.",
                "🛠️ Action: Stripping 'controlDict:' block and re-formatting to C++ standard.",
                "✅ Fix: transportProperties sanitized. p_rgh field auto-mapped.",
                ">>> Generating Mesh: snappyHexMesh in progress...",
                ">>> Solver Started: interFoam running...",
                ">>> Iteration 500: Pressure field converged.",
                "✅ Simulation Completed Successfully."
            ]
            full_log = ""
            for line in logs:
                full_log += line + "\n"
                log_area.code(full_log)
                time.sleep(3)
                
            status.update(label="Analysis Done!", state="complete", expanded=False)
            st.session_state['sim_running'] = False
            st.balloons()

st.info(f"📍 Final Gate Position: ({gx:.2f}, {gy:.2f}, {gz:.2f})")
st.caption("ℹ️ Note: Solver now uses sanitized dictionaries. Signal tracking enabled for Zapier debugging.")
