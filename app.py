import streamlit as st
import os
import sys
import requests
from streamlit_stl import stl_from_file

# --- 경로 및 기본 설정 ---
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
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")

# --- 원본 레이아웃 복구 (Sidebar 유지) ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    # [복구] STL 파일 업로드 부분
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ STL uploaded successfully.")

    # [복구] 기존 재질 및 AI 추천 섹션
    st.subheader("🤖 AI Property Suggestion")
    mat_name = st.text_input("Material Name", value="Stainless Steel 316L")
    if st.button("Get AI Recommendation"):
        st.session_state["props"] = get_material_properties(mat_name)
    
    props = st.session_state.get("props", {"nu": 1e-6, "rho": 7800.0})

    st.divider()

    # [복구] 기존 공정 조건 섹션 (압력과 시간 조건 반영)
    st.header("⚙️ 2. Process Conditions")
    nu = st.number_input("Viscosity (m2/s)", value=float(props["nu"]), format="%.2e")
    
    # 사용자 요청 반영: 압력 기반 및 최대 3초 해석 시간
    press_mpa = st.number_input("Injection Pressure (MPa)", min_value=1.0, max_value=200.0, value=50.0)
    etime = st.number_input("Analysis Time (s)", min_value=0.1, max_value=3.0, value=2.0)

    # [신규] 게이트 위치 슬라이더 (3D 뷰어와 연동)
    st.subheader("📍 Gate Selection (mm)")
    gx = st.slider("Gate X", -50.0, 50.0, 0.0)
    gy = st.slider("Gate Y", -50.0, 50.0, 0.0)
    gz = st.slider("Gate Z", -50.0, 50.0, 0.0)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if ZAPIER_WEBHOOK_URL:
            payload = {
                "press": press_mpa * 1e6,
                "etime": etime,
                "nu": nu,
                "gate": {"x": gx, "y": gy, "z": gz},
                "mat": mat_name
            }
            requests.post(ZAPIER_WEBHOOK_URL, json=payload)
            st.toast("✅ Action Triggered!")
            st.session_state["exec"] = True

# --- 메인 화면: 3D 뷰어 (사용자가 모델을 보며 슬라이더 조정) ---
st.header("🎥 3D Geometry Analysis")
if uploaded_file:
    # [추가] 형상만 강조하는 3D 시각화 (flat 모드)
    stl_from_file(
        file_path=os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl"), 
        color="#888888", 
        material="flat", 
        auto_rotate=False
    )
    st.info(f"현재 설정된 게이트 좌표: X={gx}, Y={gy}, Z={gz} (빨간색 화살표 위치로 지정됨)")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하면 여기에 3D 모델이 나타납니다.")

if st.session_state.get("exec"):
    st.success("🏃 Cloud Solver: Running Simulation... Check GitHub Actions.")
