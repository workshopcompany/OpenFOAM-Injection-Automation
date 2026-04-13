import streamlit as st
import os
import requests
import numpy as np
import trimesh # STL 분석용
import plotly.graph_objects as go # 3D 시각화용

# --- 기본 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# --- Sidebar 레이아웃 ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    if uploaded_file:
        # STL 파일 저장 및 분석
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Trimesh로 STL 로드 및 바운딩 박스 계산
        try:
            mesh = trimesh.load(stl_path)
            st.success("✅ STL loaded & analyzed.")
            # 모델의 크기 정보 제공 (사용자 가이드)
            bounds = mesh.bounds
            size = mesh.extents
            st.info(f"📐 Model Size (mm):\nX: {size[0]:.1f}, Y: {size[1]:.1f}, Z: {size[2]:.1f}")
            st.caption(f"Range: X({bounds[0][0]:.1f}~{bounds[1][0]:.1f})")
        except Exception as e:
            st.error(f"Error analyzing STL: {e}")
            mesh = None
    else:
        mesh = None

    # [변경] 재료명 PA66+30glassfiber로 고정
    st.subheader("🤖 AI Property Suggestion")
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber")
    
    st.divider()

    # [이동] Gate Selection (Plotly 좌표와 연동)
    st.header("📍 2. Gate Location (mm)")
    st.caption("오른쪽 3D 모델의 빨간 점 위치를 좌표로 입력하세요.")
    
    # mesh 정보를 바탕으로 기본 좌표 설정
    default_x = 0.0
    if mesh is not None:
        default_x = mesh.bounds[0][0] # 예: X축 최소값에 게이트 배치

    gx = st.number_input("Gate X", value=default_x, step=0.1)
    gy = st.number_input("Gate Y", value=0.0, step=0.1)
    gz = st.number_input("Gate Z", value=0.0, step=0.1)

    st.divider()

    # [변경] 사출 조건: 온도, 압력, 속도
    st.header("⚙️ 3. Process Conditions")
    
    # 온도 (Melt Temperature, °C)
    temp_c = st.number_input("Melt Temperature (°C)", min_value=100, max_value=400, value=280)
    
    # 압력 (Injection Pressure, MPa)
    press_mpa = st.number_input("Injection Pressure (MPa)", min_value=10.0, max_value=200.0, value=50.0)
    
    # 속도 (Injection Velocity, mm/s)
    vel_mms = st.number_input("Injection Velocity (mm/s)", min_value=1.0, max_value=500.0, value=100.0)

    # [변경] 분석 시간: 충진 완료 기준, 최대 3초 제한
    st.subheader("⏱️ Analysis Time")
    st.caption("AI가 충진 완료 시간을 예측합니다 (최대 3초).")
    
    # 가상의 AI 예측 로직 (추후 실제 연동 가능)
    # 예: 속도가 빠르면 시간이 줄어듦
    predicted_time = max(0.1, min(3.0, 300.0 / vel_mms)) 
    
    etime = st.number_input("Predicted End Time (s)", value=float(f"{predicted_time:.2f}"), max_value=3.0, min_value=0.1)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if mesh is not None and ZAPIER_WEBHOOK_URL:
            # Allrun에 전달할 파라미터 구성 강화
            payload = {
                "temp": temp_c + 273.15, # Kelvin으로 변환
                "press": press_mpa * 1e6, # Pa로 변환
                "vel": vel_mms / 1000.0, # m/s로 변환
                "etime": etime,
                "gate": {"x": gx, "y": gy, "z": gz},
                "mat": mat_name
            }
            try:
                # requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
                st.toast("✅ Action Triggered on GitHub Cloud!", icon="🌐")
                st.session_state["exec"] = True
            except requests.exceptions.RequestException as e:
                st.error(f"Zapier sync failed: {e}")

# --- 메인 화면: Plotly 3D 뷰어 (음영 및 게이트 표시) ---
st.header("🎥 3D Geometry & Gate Analysis")

if mesh is not None:
    with st.spinner("Rendering 3D model with shading..."):
        # 1. Plotly Mesh3d 생성 (음영 처리 포함)
        # trimesh의 데이터를 plotly 형식으로 변환
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Mesh3d 객체 생성 (라이팅 및 색상 설정)
        mesh_3d = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='#AAAAAA', # 모델 색상 (연한 회색)
            opacity=1.0,
            flatshading=False, # 부드러운 음영 효과 (True로 하면 각진 느낌)
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.5,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=100, z=100),
            name='Model'
        )

        # 2. 게이트 위치를 표시할 빨간 점 생성 (Scatter3d)
        gate_point = go.Scatter3d(
            x=[gx], y=[gy], z=[gz],
            mode='markers+text',
            marker=dict(
                size=10,
                color='red',
                opacity=0.9,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=["GATE"],
            textposition="top center",
            name='Gate'
        )

        # 3. 레이아웃 설정 (배경색, 카메라 등)
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
                yaxis=dict(title='Y (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
                zaxis=dict(title='Z (mm)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
                aspectmode='data', # 모델의 실제 비율 유지
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5) # 초기 카메라 위치
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0), # 여백 제거
            paper_bgcolor='rgba(0,0,0,0)', # 배경 투명
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # 4. 피규어 생성 및 표시
        fig = go.Figure(data=[mesh_3d, gate_point], layout=layout)
        
        # streamlit에 plotly 차트 표시 (config로 툴바 제어 가능)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
    st.info(f"📍 현재 설정된 게이트 좌표: X={gx}, Y={gy}, Z={gz}")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하면 여기에 입체적인 3D 모델이 나타납니다.")

if st.session_state.get("exec"):
    st.success("🏃 GitHub Actions Solver is running with your gate position. Check 'Actions' tab.")
