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

# CSS를 이용해 화면 중앙에 빨간 점과 사각 가이드라인 생성
st.markdown("""
    <style>
    .viewer-container {
        position: relative;
        border: 2px solid #555; /* 사각형 표시 */
        border-radius: 10px;
    }
    .red-dot {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 12px;
        height: 12px;
        background-color: red;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        z-index: 10;
        box-shadow: 0 0 10px white;
        pointer-events: none; /* 클릭 방해 금지 */
    }
    .guide-text {
        position: absolute;
        bottom: 10px;
        right: 10px;
        color: #ff4b4b;
        font-weight: bold;
        z-index: 11;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 MIM-Ops: Cloud Simulation")

# --- Sidebar 레이아웃 (순서 변경 적용) ---
with st.sidebar:
    st.header("📂 1. Geometry & Gate")
    
    # [복구] STL 파일 업로드
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ STL uploaded.")

    # [이동] Gate Selection UI를 AI Suggestion 위로 올림
    st.subheader("📍 Gate Selection (mm)")
    st.caption("화면 중앙의 빨간 점에 게이트 위치를 맞추고 좌표를 입력하세요.")
    gx = st.number_input("Gate X", value=0.0, step=0.1)
    gy = st.number_input("Gate Y", value=0.0, step=0.1)
    gz = st.number_input("Gate Z", value=0.0, step=0.1)

    st.divider()

    # [이동] AI Property Suggestion (순서 변경)
    st.subheader("🤖 AI Property Suggestion")
    # [변경] 기본 재료명을 PA66+30glassfiber로 수정
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber")
    if st.button("Get AI Recommendation"):
        st.session_state["props"] = get_material_properties(mat_name)
    
    props = st.session_state.get("props", {"nu": 1e-6, "rho": 1350.0}) # PA66 GF30 밀도 근사치

    st.divider()

    # [유지] 기존 공정 조건 섹션
    st.header("⚙️ 2. Process Conditions")
    nu = st.number_input("Viscosity (m2/s)", value=float(props["nu"]), format="%.2e")
    press_mpa = st.number_input("Injection Pressure (MPa)", min_value=1.0, max_value=200.0, value=50.0)
    etime = st.number_input("Analysis Time (s)", min_value=0.1, max_value=3.0, value=2.0)

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

# --- 메인 화면: 3D 뷰어 + 조준점 시스템 ---
st.header("🎥 3D Geometry Analysis")

if uploaded_file:
    # 뷰어와 빨간 점을 감싸는 컨테이너 시작
    st.markdown('<div class="viewer-container">', unsafe_allow_html=True)
    st.markdown('<div class="red-dot"></div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-text">TARGET GATE POINT</div>', unsafe_allow_html=True)
    
    stl_from_file(
        file_path=os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl"), 
        color="#888888", 
        material="flat", 
        auto_rotate=False
    )
    
    st.markdown('</div>', unsafe_allow_html=True) # 컨테이너 종료
    st.info(f"조준점에 맞춘 현재 설정 좌표: X={gx}, Y={gy}, Z={gz}")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하세요.")

if st.session_state.get("exec"):
    st.success("🏃 Cloud Solver: Running Simulation...")
