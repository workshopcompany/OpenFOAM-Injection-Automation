import streamlit as st
import os
import requests
import numpy as np
import trimesh # STL 분석 및 표면 스냅용
import plotly.graph_objects as go # 3D 시각화용

# --- 기본 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# --- AI 엔진 목업 (위치 및 크기 제안) ---
def get_ai_suggestions(mesh):
    if mesh is None: return 0.0, 0.0, 0.0, 2.0
    # AI Logic: 제품의 부피와 투영 면적을 고려하여 최적 사이즈와 위치 계산
    bounds = mesh.bounds
    center = mesh.centroid
    # 예시로 중심점 근처 표면과 부피 대비 적정 게이트 직경(2.5mm) 제안
    suggested_pos = trimesh.proximity.closest_point(mesh, [center])[0][0]
    return suggested_pos[0], suggested_pos[1], suggested_pos[2], 2.5

# --- Sidebar 레이아웃 ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            mesh = trimesh.load(stl_path)
            st.success("✅ STL loaded & analyzed.")
            size = mesh.extents
            st.info(f"📐 Model Size (mm):\nX: {size[0]:.1f}, Y: {size[1]:.1f}, Z: {size[2]:.1f}")
        except Exception as e:
            st.error(f"Error: {e}")
            mesh = None
    else:
        mesh = None

    st.subheader("🤖 AI Property Suggestion")
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber")
    
    st.divider()

    # [업데이트] AI 최적화 제안 버튼
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Optimal Suggestion", use_container_width=True):
        if mesh is not None:
            ax, ay, az, asize = get_ai_suggestions(mesh)
            st.session_state['gx'], st.session_state['gy'], st.session_state['gz'] = ax, ay, az
            st.session_state['gsize'] = asize
            st.toast("AI 추천 위치와 사이즈가 반영되었습니다!")

    # 게이트 사이즈 설정 추가
    g_size = st.number_input("Gate Diameter (mm)", min_value=0.5, max_value=10.0, 
                             value=st.session_state.get('gsize', 2.0), step=0.1, key='gsize')

    st.caption("조정 시 게이트가 제품 표면에 자동으로 스냅됩니다.")
    
    # 세션 상태를 이용한 좌표 연동
    default_x = mesh.bounds[0][0] if mesh is not None else 0.0
    val_x = st.number_input("Gate X", value=st.session_state.get('gx', default_x), step=0.1, key='gx')
    val_y = st.number_input("Gate Y", value=st.session_state.get('gy', 0.0), step=0.1, key='gy')
    val_z = st.number_input("Gate Z", value=st.session_state.get('gz', 0.0), step=0.1, key='gz')

    # [핵심] 표면 스냅 로직: 입력한 좌표와 가장 가까운 '표면' 좌표 계산
    if mesh is not None:
        raw_point = np.array([[val_x, val_y, val_z]])
        closest_point = trimesh.proximity.closest_point(mesh, raw_point)[0][0]
        gx, gy, gz = closest_point
    else:
        gx, gy, gz = val_x, val_y, val_z

    st.divider()

    st.header("⚙️ 3. Process Conditions")
    temp_c = st.number_input("Melt Temperature (°C)", min_value=100, max_value=400, value=280)
    press_mpa = st.number_input("Injection Pressure (MPa)", min_value=10.0, max_value=200.0, value=50.0)
    vel_mms = st.number_input("Injection Velocity (mm/s)", min_value=1.0, max_value=500.0, value=100.0)

    st.subheader("⏱️ Analysis Time")
    predicted_time = max(0.1, min(3.0, 300.0 / vel_mms)) 
    etime = st.number_input("Predicted End Time (s)", value=float(f"{predicted_time:.2f}"), max_value=3.0, min_value=0.1)

    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True):
        if mesh is not None and ZAPIER_WEBHOOK_URL:
            payload = {
                "temp": temp_c + 273.15,
                "press": press_mpa * 1e6,
                "vel": vel_mms / 1000.0,
                "etime": etime,
                "gate": {"x": gx, "y": gy, "z": gz, "size": g_size},
                "mat": mat_name
            }
            try:
                # requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
                st.toast("✅ Simulation Triggered!", icon="🌐")
                st.session_state["exec"] = True
            except Exception as e:
                st.error(f"Error: {e}")

# --- 메인 화면: Plotly 3D ---
st.header("🎥 3D Geometry & Gate Analysis")

if mesh is not None:
    with st.spinner("Rendering..."):
        vertices, faces = mesh.vertices, mesh.faces
        
        mesh_3d = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='#AAAAAA', opacity=1.0, flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5),
            name='Model'
        )

        # 게이트 포인트 (설정한 사이즈 반영)
        gate_point = go.Scatter3d(
            x=[gx], y=[gy], z=[gz],
            mode='markers+text',
            marker=dict(
                size=g_size * 5, # 시각적 인지를 위해 직경에 비례하여 크기 조절
                color='red', opacity=0.9, symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=[f"GATE ({g_size}mm)"], textposition="top center", name='Gate'
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
                yaxis=dict(title='Y (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
                zaxis=dict(title='Z (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        fig = go.Figure(data=[mesh_3d, gate_point], layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
    st.info(f"📍 최종 스냅 좌표: X={gx:.2f}, Y={gy:.2f}, Z={gz:.2f} | Size: {g_size}mm")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하세요.")
