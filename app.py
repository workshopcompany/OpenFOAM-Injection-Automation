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

# --- AI 통합 제안 함수 ---
def get_ai_comprehensive_suggestions(mesh, material_name):
    pos = [0.0, 0.0, 0.0]
    g_size = 2.0
    temp, press, vel = 280, 50, 100
    
    if mesh is not None:
        # 위치 제안: 제품 중심 근처 표면
        center = mesh.centroid
        pos = trimesh.proximity.closest_point(mesh, [center])[0][0]
        
        # 재료별 공정 조건 분기
        if "PA66" in material_name.upper():
            temp, press, vel, g_size = 290, 80, 120, 2.5
        else:
            temp, press, vel, g_size = 250, 60, 100, 2.0
            
    return pos, g_size, temp, press, vel

# --- Sidebar 레이아웃 ---
with st.sidebar:
    # --- 1. Geometry ---
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
            size = mesh.extents
            st.info(f"📏 Size: X:{size[0]:.1f}, Y:{size[1]:.1f}, Z:{size[2]:.1f}")
        except: st.error("STL 분석 실패")

    st.divider()

    # --- 2. Gate Configuration ---
    st.header("📍 2. Gate Configuration")
    
    # 통합 AI 제안 버튼 (Material Name을 참고하여 제안하므로 순서상 아래에 있어도 내부 로직 연동)
    if st.button("🪄 AI Comprehensive Suggestion", use_container_width=True):
        if mesh is not None:
            # 현재 입력된 재료명을 가져옴
            current_mat = st.session_state.get('mat_name_input', "PA66+30glassfiber")
            pos, g_size, t, p, v = get_ai_comprehensive_suggestions(mesh, current_mat)
            st.session_state['gx'], st.session_state['gy'], st.session_state['gz'] = pos
            st.session_state['gsize'] = g_size
            st.session_state['temp'] = t
            st.session_state['press'] = p
            st.session_state['vel'] = v
            st.toast("AI 추천 설정이 반영되었습니다!")

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, st.session_state.get('gsize', 2.0), key='gsize')
    
    # 좌표 입력 및 스냅
    def_x = mesh.bounds[0][0] if mesh is not None else 0.0
    vx = st.number_input("Gate X", value=st.session_state.get('gx', def_x), key='gx', step=0.1)
    vy = st.number_input("Gate Y", value=st.session_state.get('gy', 0.0), key='gy', step=0.1)
    vz = st.number_input("Gate Z", value=st.session_state.get('gz', 0.0), key='gz', step=0.1)

    if mesh is not None:
        gx, gy, gz = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])[0][0]
    else:
        gx, gy, gz = vx, vy, vz

    st.divider()

    # --- 3. Material ---
    st.header("🧪 3. Material")
    # key를 부여하여 AI 제안 시에도 연동되도록 설정
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber", key='mat_name_input')
    
    st.divider()

    # --- 4. Process Condition ---
    st.header("⚙️ 4. Process Condition")
    temp_c = st.number_input("Melt Temperature (°C)", 100, 400, st.session_state.get('temp', 280), key='temp')
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 200.0, st.session_state.get('press', 50.0), key='press')
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 500.0, st.session_state.get('vel', 100.0), key='vel')

    st.subheader("⏱️ Analysis Time")
    predicted_time = max(0.1, min(3.0, 300.0 / vel_mms)) 
    etime = st.number_input("End Time (s)", value=float(f"{predicted_time:.2f}"), max_value=3.0, key='etime')

    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True):
        st.toast("✅ Action Triggered!", icon="🌐")

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
    # 게이트 시각화 (빨간 점)
    gate_point = go.Scatter3d(
        x=[gx], y=[gy], z=[gz], mode='markers',
        marker=dict(size=g_size*5, color='red', opacity=0.9, line=dict(color='white', width=2))
    )
    fig = go.Figure(data=[mesh_3d, gate_point])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"📍 Final Snapped Gate: ({gx:.2f}, {gy:.2f}, {gz:.2f})")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하세요.")
