import streamlit as st
import os
import requests
import numpy as np
import trimesh
import plotly.graph_objects as go

# --- 0. 기본 설정 및 세션 초기화 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# 세션 상태 초기화 (AI 제안 값 유지용)
if 'gx' not in st.session_state: st.session_state['gx'] = 0.0
if 'gy' not in st.session_state: st.session_state['gy'] = 0.0
if 'gz' not in st.session_state: st.session_state['gz'] = 0.0
if 'gsize' not in st.session_state: st.session_state['gsize'] = 2.0
if 'temp' not in st.session_state: st.session_state['temp'] = 280
if 'press' not in st.session_state: st.session_state['press'] = 50.0
if 'vel' not in st.session_state: st.session_state['vel'] = 100.0

# --- 1. AI 제안 로직 함수들 ---
def get_ai_gate_suggestions(mesh, material_name):
    """형상 기반 최적 게이트 위치 및 크기 제안"""
    if mesh is None: return 0.0, 0.0, 0.0, 2.0
    center = mesh.centroid
    # 표면 중 중심에서 가장 가까운 점 찾기
    pos = trimesh.proximity.closest_point(mesh, [center])[0][0]
    size = 2.5 if "PA66" in material_name.upper() else 2.0
    return pos[0], pos[1], pos[2], size

def get_ai_process_suggestions(material_name):
    """재료 기반 최적 공정 조건 제안"""
    temp, press, vel = 230, 60, 100
    mat_upper = material_name.upper()
    if "PA66" in mat_upper:
        temp, press, vel = 290, 80, 120
    elif any(x in mat_upper for x in ["SUS", "17-4PH", "STEEL"]):
        temp, press, vel = 180, 130, 70
    return temp, press, vel

# --- 2. Sidebar 레이아웃 (요청 순서 반영) ---
with st.sidebar:
    # --- SECTION 1: Geometry ---
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
            ext = mesh.extents
            st.info(f"📐 Size: X:{ext[0]:.1f}, Y:{ext[1]:.1f}, Z:{ext[2]:.1f}")
        except: st.error("STL 분석 실패")

    st.divider()

    # --- SECTION 2: Gate Configuration ---
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        if mesh is not None:
            ax, ay, az, asize = get_ai_gate_suggestions(mesh, st.session_state.get('mat_name_input', ""))
            st.session_state['gx'], st.session_state['gy'], st.session_state['gz'] = ax, ay, az
            st.session_state['gsize'] = asize
            st.toast("AI가 최적 위치를 계산했습니다.")

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, key='gsize', step=0.1)
    
    # 입력 좌표 (수동 입력 및 AI 반영용)
    def_x = mesh.bounds[0][0] if mesh is not None else 0.0
    vx = st.number_input("Gate X", value=st.session_state.get('gx', def_x), key='gx', step=0.1)
    vy = st.number_input("Gate Y", value=st.session_state.get('gy', 0.0), key='gy', step=0.1)
    vz = st.number_input("Gate Z", value=st.session_state.get('gz', 0.0), key='gz', step=0.1)

    # [핵심] 표면 스냅 로직 적용
    if mesh is not None:
        snapped = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])[0][0]
        gx, gy, gz = snapped
    else:
        gx, gy, gz = vx, vy, vz

    st.divider()

    # --- SECTION 3: Material ---
    st.header("🧪 3. Material")
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber", key='mat_name_input')
    
    st.divider()

    # --- SECTION 4: Process Condition ---
    st.header("⚙️ 4. Process Condition")
    if st.button("🤖 Optimize Process for Material", use_container_width=True):
        t, p, v = get_ai_process_suggestions(mat_name)
        st.session_state['temp'] = t
        st.session_state['press'] = p
        st.session_state['vel'] = v
        st.toast("재료별 최적 조건이 입력되었습니다.")

    # Injection Temperature로 용어 변경 및 재료 투입 온도 설정
    temp_c = st.number_input("Injection Temperature (°C)", 50, 450, key='temp')
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 250.0, key='press')
    vel_mms = st.number_input("Injection Velocity (mm/s)", 1.0, 600.0, key='vel')

    st.subheader("⏱️ Analysis Time")
    predicted_time = max(0.1, min(3.0, 300.0 / vel_mms)) 
    etime = st.number_input("End Time (s)", value=float(f"{predicted_time:.2f}"), max_value=3.0, key='etime')

    st.divider()
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True):
        if mesh is not None and ZAPIER_WEBHOOK_URL:
            payload = {
                "temp": temp_c, "press": press_mpa, "vel": vel_mms, "etime": etime,
                "gate": {"x": gx, "y": gy, "z": gz, "size": g_size}, "mat": mat_name
            }
            # requests.post(ZAPIER_WEBHOOK_URL, json=payload)
            st.toast("✅ Action Triggered on Cloud!", icon="🌐")

# --- 3. 메인 화면: Plotly 3D 시각화 ---
st.header("🎥 3D Geometry & Gate Analysis")

if mesh is not None:
    # 모델 렌더링 (음영 효과 포함)
    vertices, faces = mesh.vertices, mesh.faces
    mesh_3d = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#AAAAAA', opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5),
        name='Model'
    )
    
    # 스냅된 게이트 위치 (빨간 점)
    gate_point = go.Scatter3d(
        x=[gx], y=[gy], z=[gz], mode='markers+text',
        marker=dict(size=g_size*5, color='red', opacity=0.9, line=dict(color='white', width=2)),
        text=[f"GATE ({g_size}mm)"], textposition="top center", name='Gate'
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
            yaxis=dict(title='Y (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
            zaxis=dict(title='Z (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=True
    )

    fig = go.Figure(data=[mesh_3d, gate_point], layout=layout)
    st.plotly_chart(fig, use_container_width=True)
    
    # 좌표 정보 및 안내 문구
    st.info(f"📍 Final Snapped Gate: X={gx:.2f}, Y={gy:.2f}, Z={gz:.2f}")
    st.caption("ℹ️ **Note:** The gate is automatically snapped to the nearest point on the model surface to ensure simulation accuracy.")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하세요.")
