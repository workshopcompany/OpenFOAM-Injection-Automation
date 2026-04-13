import streamlit as st
import os
import numpy as np
import trimesh
import plotly.graph_objects as go

# --- 기본 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# --- AI 통합 제안 함수 (위치, 크기, 공정 조건) ---
def get_ai_comprehensive_suggestions(mesh, material_name):
    # 기본값 설정
    pos = [0.0, 0.0, 0.0]
    g_size = 2.0
    temp, press, vel = 280, 50, 100
    
    if mesh is not None:
        # 1. 위치 제안: 제품의 무게 중심 근처 표면
        center = mesh.centroid
        pos = trimesh.proximity.closest_point(mesh, [center])[0][0]
        
        # 2. 재료/형상 기반 공정 조건 제안 (PA66+30GF 기준 로직)
        if "PA66" in material_name.upper():
            temp = 290   # PA66 GF30 권장 수지 온도
            press = 80   # 충진 압력 제안
            vel = 120    # 사출 속도 제안
            g_size = 2.5 # 섬유 함유 재료 특성상 조금 큰 게이트
        else:
            temp, press, vel, g_size = 250, 60, 100, 2.0
            
    return pos, g_size, temp, press, vel

# --- Sidebar 레이아웃 ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
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

    # [재료 선정]
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber")
    
    st.divider()

    # [변경] 2번 섹션을 Gate Configuration으로 올림
    st.header("📍 2. Gate Configuration")
    
    # 통합 AI 제안 버튼 (위치 + 사이즈 + 공정조건 한꺼번에)
    if st.button("🪄 AI Comprehensive Suggestion", use_container_width=True):
        if mesh is not None:
            pos, g_size, t, p, v = get_ai_comprehensive_suggestions(mesh, mat_name)
            st.session_state['gx'], st.session_state['gy'], st.session_state['gz'] = pos
            st.session_state['gsize'] = g_size
            st.session_state['temp'] = t
            st.session_state['press'] = p
            st.session_state['vel'] = v
            st.toast("AI가 최적의 게이트와 공정 조건을 제안했습니다!", icon="🤖")

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, st.session_state.get('gsize', 2.0), key='gsize')
    
    # 좌표 입력 (스냅 로직 포함)
    def_x = mesh.bounds[0][0] if mesh is not None else 0.0
    vx = st.number_input("Gate X", value=st.session_state.get('gx', def_x), key='gx')
    vy = st.number_input("Gate Y", value=st.session_state.get('gy', 0.0), key='gy')
    vz = st.number_input("Gate Z", value=st.session_state.get('gz', 0.0), key='gz')

    if mesh is not None:
        gx, gy, gz = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])[0][0]
    else:
        gx, gy, gz = vx, vy, vz

    st.divider()

    # [변경] 3번 섹션: 공정 조건 (AI 제안 반영)
    st.header("⚙️ 3. Process Conditions")
    
    # AI가 제안한 값이 있으면 반영, 없으면 기본값
    temp_c = st.number_input("Melt Temperature (°C)", 100, 400, st.session_state.get('temp', 280), key='temp')
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 200.0, st.session_state.get('press', 50.0), key='press')
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 500.0, st.session_state.get('vel', 100.0), key='vel')

    st.subheader("⏱️ Analysis Time")
    predicted_time = max(0.1, min(3.0, 300.0 / vel_mms)) 
    etime = st.number_input("Predicted End Time (s)", value=float(f"{predicted_time:.2f}"), max_value=3.0)

    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True):
        st.toast("✅ Simulation Triggered!", icon="🌐")

# --- 메인 화면: Plotly 3D ---
st.header("🎥 3D Geometry & Gate Analysis")

if mesh is not None:
    vertices, faces = mesh.vertices, mesh.faces
    mesh_3d = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#AAAAAA', opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5)
    )
    gate_point = go.Scatter3d(
        x=[gx], y=[gy], z=[gz], mode='markers',
        marker=dict(size=g_size*5, color='red', opacity=0.9, line=dict(color='white', width=2))
    )
    fig = go.Figure(data=[mesh_3d, gate_point])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"📍 Final Gate: ({gx:.2f}, {gy:.2f}, {gz:.2f}) | Size: {g_size}mm")
