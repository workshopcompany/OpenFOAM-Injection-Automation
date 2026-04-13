import streamlit as st
import os
import trimesh # STL 분석 및 표면 거리 계산용
import plotly.graph_objects as go # 3D 시각화
import numpy as np

# --- 기본 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro: Gate Snap", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# --- 1. AI 최적 게이트 위치 제안 함수 (가상의 AI 엔진 목업) ---
def get_ai_gate_suggestion(mesh):
    """
    AI가 형상을 분석하여 최적의 게이트 위치(X, Y, Z)를 제안합니다.
    (현재는 시각적 시연을 위해 Bounding Box의 한 점을 제안하는 목업입니다.)
    """
    if mesh is None: return 0, 0, 0
    
    # [예시] AI logic: "제품의 가장 긴 축 끝단 중앙"을 제안
    bounds = mesh.bounds
    # 예: X축 최소값, Y축 중앙, Z축 중앙
    suggested_x = bounds[0][0]
    suggested_y = (bounds[0][1] + bounds[1][1]) / 2.0
    suggested_z = (bounds[0][2] + bounds[1][2]) / 2.0
    
    # 중요: 이 제안된 좌표도 반드시 제품 표면에 있어야 합니다.
    # 근접 표면 좌표를 계산하는 내부 함수(아래)를 활용하여 정제
    refined_suggestion = trimesh.proximity.closest_point(mesh, [[suggested_x, suggested_y, suggested_z]])
    
    return refined_suggestion[0][0] # 정제된 실제 표면 좌표($x, y, z$) 반환

# --- 2. [가칭] 표면 자석 스냅 함수 ---
def snap_gate_to_surface(mesh, target_coord):
    """
    사용자가 입력한 목표 좌표와 가장 가까운 제품 표면의 실제 좌표를 계산합니다.
    """
    if mesh is None: return target_coord
    
    target_point = np.array([target_coord])
    
    # trimesh의 proximity 엔진을 사용하여 가장 가까운 표면 점 계산
    closest_point, distance, face_id = trimesh.proximity.closest_point(mesh, target_point)
    
    return closest_point[0] # 표면에 '스냅'된 실제 좌표 반환

# --- Sidebar 레이아웃 ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            mesh = trimesh.load(stl_path)
            st.success("✅ STL loaded.")
            size = mesh.extents
            st.info(f"📐 Model Size: X:{size[0]:.1f}, Y:{size[1]:.1f}, Z:{size[2]:.1f}")
        except: mesh = None
    else: mesh = None

    st.subheader("🤖 AI Property Suggestion")
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber")
    
    st.divider()

    # [업데이트] Gate Location: AI 제안 및 스냅 기능 통합
    st.header("📍 2. Gate Location (mm)")
    
    # 1. 1차 AI 최적 위치 제안 버튼
    if uploaded_file and st.button("🤖 Get AI suggestion for optimal gate"):
        with st.spinner("AI analyzing geometry..."):
            ai_x, ai_y, ai_z = get_ai_gate_suggestion(mesh)
            # AI가 제안한 표면 좌표를 session_state에 저장
            st.session_state["gate_coord"] = {"x": ai_x, "y": ai_y, "z": ai_z}
            st.toast("AI 추천 좌표가 반영되었습니다.", icon="🤖")

    # [수정] Session State를 이용한 좌표 관리 (AI 제안 또는 수동 입력 유지)
    gate = st.session_state.get("gate_coord", {"x": 0.0, "y": 0.0, "z": 0.0})
    
    # 2. 마우스 입력 슬라이더 (사용자는 대략 움직임)
    st.caption("좌표를 조정하세요. 빨간 점은 자동으로 제품 표면에 달라붙습니다.")
    raw_gx = st.slider("Target X", -100.0, 100.0, float(gate["x"]))
    raw_gy = st.slider("Target Y", -100.0, 100.0, float(gate["y"]))
    raw_gz = st.slider("Target Z", -100.0, 100.0, float(gate["z"]))
    
    # [핵심] 표면 자석 스냅 로직 적용
    # 사용자가 입력한 raw_ 좌표를 가장 가까운 표면 좌표로 변환
    snapped_coord = snap_gate_to_surface(mesh, [raw_gx, raw_gy, raw_gz])
    gx, gy, gz = snapped_coord[0], snapped_coord[1], snapped_coord[2]
    
    # 최종 스냅된 좌표 표시
    st.success(f"Final Snapped Gate: ({gx:.1f}, {gy:.1f}, {gz:.1f})")

    st.divider()

    # 사출 조건 및 분석 시간 (최대 3초) 섹션은 기존 코드 유지
    st.header("⚙/U 3. Process Conditions")
    press_mpa = st.number_input("Injection Pressure (MPa)", value=50.0)
    etime = st.slider("Analysis Time (s)", 0.5, 3.0, 2.0, 0.1)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        # 파라미터 전달 로직은 기존 코드 유지
        pass

# --- 메인 화면: Plotly 3D (음영 처리 및 스냅된 게이트 표시) ---
st.header("🎥 3D Geometry & Gate Analysis")

if mesh is not None:
    # Plotly Mesh3d 생성 (음영 처리 포함)
    vertices = mesh.vertices
    faces = mesh.faces
    
    mesh_3d = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#AAAAAA', opacity=1.0, flatshading=False, lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5), name='Model'
    )

    # [업데이트] 스냅된 게이트 위치를 표시할 빨간 점 생성 (Scatter3d)
    gate_point = go.Scatter3d(
        x=[gx], y=[gy], z=[gz], # 최종 스냅된 좌표 사용
        mode='markers+text',
        marker=dict(size=12, color='red', opacity=0.9, line=dict(color='white', width=2)),
        text=["GATE"], textposition="top center", name='Gate'
    )

    # 레이아웃 설정
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X (mm)'), yaxis=dict(title='Y (mm)'), zaxis=dict(title='Z (mm)'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True
    )

    fig = go.Figure(data=[mesh_3d, gate_point], layout=layout)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하세요.")
