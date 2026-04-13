import streamlit as st
import os
import sys
import requests
from streamlit_stl import stl_from_file

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# AI Advisor 로드
try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'scripts/gemini_advisor.py' missing.")

ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI 설정 ---
st.set_page_config(page_title="MIM-Ops Pro: 3D Visualizer", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered 3D Analysis")

# 좌우 레이아웃 분할 (왼쪽: 설정, 오른쪽: 3D 뷰어)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📂 1. Geometry & Gate")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    if uploaded_file:
        # 파일 임시 저장
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ STL Loaded. Analyze the 3D model on the right.")

    # AI 게이트 위치 제안 버튼
    if st.button("🤖 AI Gate Suggestion"):
        with st.spinner("Analyzing geometry for optimal flow..."):
            # 추후 AI 좌표 연동 로직이 들어갈 자리입니다.
            st.info("AI 추천: 모델의 하단 중앙부(Z=0)를 게이트로 추천합니다.")
            st.session_state["gate_coord"] = {"x": 0.0, "y": 0.0, "z": 0.0}

    # 수동 게이트 좌표 입력
    gate = st.session_state.get("gate_coord", {"x": 0.0, "y": 0.0, "z": 0.0})
    st.write("Current Gate Target:")
    gc1, gc2, gc3 = st.columns(3)
    gx = gc1.number_input("X", value=gate["x"])
    gy = gc2.number_input("Y", value=gate["y"])
    gz = gc3.number_input("Z", value=gate["z"])

    st.divider()
    
    st.header("⚙️ 2. Process Conditions")
    # 압력 제어 조건 (사용자 의견 반영: MPa 단위)
    press_mpa = st.number_input("Injection Pressure (MPa)", min_value=1.0, max_value=300.0, value=50.0)
    
    # 해석 시간 제어 (사용자 의견 반영: 최대 3초)
    etime = st.slider("Analysis Time (s)", 0.1, 3.0, 2.0, 0.1)
    
    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if ZAPIER_WEBHOOK_URL:
            payload = {
                "press": press_mpa * 1e6, # Pa로 변환하여 전달
                "etime": etime,
                "gate": {"x": gx, "y": gy, "z": gz},
                "mat": st.session_state.get("mat_name", "Unknown")
            }
            requests.post(ZAPIER_WEBHOOK_URL, json=payload)
            st.toast("✅ Simulation Triggered!")
            st.session_state["exec"] = True

with col2:
    st.header("🎥 3D Geometry Preview")
    if uploaded_file:
        # STL 뷰어 실행 (색상 및 회전 설정 가능)
        stl_from_file(file_path=stl_path, color="#4CAF50", material="plastic", auto_rotate=True)
        st.caption("3D 뷰어를 통해 형상을 회전하며 게이트 위치를 확인하세요.")
    else:
        st.info("파일을 업로드하면 여기에 3D 모델이 나타납니다.")

# --- 상태 로그 표시 ---
if st.session_state.get("exec"):
    st.success("🏃 GitHub Cloud Solver: Running Simulation...")
    st.code(f">>> Target: {press_mpa} MPa | Duration: {etime}s\n>>> Gate: ({gx}, {gy}, {gz})")
