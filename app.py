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

# AI Advisor 로드 (경로 에러 방지)
try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'scripts/gemini_advisor.py' missing.")
    # 임시 목업 함수 (없을 경우 대비)
    def get_material_properties(name): return {"nu": 1e-6, "rho": 1350.0}

ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI 설정 ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")

# CSS 수정: 사각형 테두리를 지우고 조준점 위치를 아래로 내림
# 핵심: 컨테이너 높이를 고정(height: 500px;)하여 absolute 계산을 안정화
st.markdown("""
    <style>
    .viewer-container {
        position: relative;
        width: 100%;
        height: 500px; /* 3D 모델이 나타날 고정 높이 */
        overflow: hidden;
        margin-top: 10px;
    }
    .red-dot {
        position: absolute;
        top: 50%; /* 컨테이너 중앙(250px 지점)으로 내림 */
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
    /* "TARGET GATE POINT" 텍스트를 빨간 점 바로 오른쪽에 배치 */
    .guide-text {
        position: absolute;
        top: calc(50% + 15px); /* 점 아래 15px */
        left: 50%;
        transform: translateX(-50%); /* 중앙 정렬 */
        color: #ff4b4b;
        font-size: 14px;
        font-weight: bold;
        z-index: 11;
        background-color: rgba(0,0,0,0.5); /* 검은 배경에 읽기 편하게 */
        padding: 2px 5px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 MIM-Ops: Cloud Simulation")

# --- Sidebar 레이아웃 (순서 및 기본 재료 PA66 GF30 적용) ---
with st.sidebar:
    st.header("📂 1. Geometry & Gate")
    
    # [유지] STL 파일 업로드
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    
    # [유지] Gate Selection UI (AI Suggestion 위)
    st.subheader("📍 Gate Selection (mm)")
    st.caption("화면 중앙의 빨간 점에 게이트 위치를 맞추고 좌표를 입력하세요.")
    gx = st.number_input("Gate X", value=0.0, step=0.1)
    gy = st.number_input("Gate Y", value=0.0, step=0.1)
    gz = st.number_input("Gate Z", value=0.0, step=0.1)

    st.divider()

    # [유지] AI Property Suggestion (순서 변경)
    st.subheader("🤖 AI Property Suggestion")
    # [변경] 기본 재료명을 PA66+30glassfiber로 수정
    mat_name = st.text_input("Material Name", value="PA66+30glassfiber")
    if st.button("Get AI Recommendation"):
        st.session_state["props"] = get_material_properties(mat_name)
    
    # PA66 GF30 근사 밀도
    props = st.session_state.get("props", {"nu": 1e-6, "rho": 1350.0})

    st.divider()

    # [유지] 기존 공정 조건 섹션
    st.header("⚙️ 2. Process Conditions")
    nu = st.number_input("Viscosity (m2/s)", value=float(props["nu"]), format="%.2e")
    press_mpa = st.number_input("Injection Pressure (MPa)", min_value=1.0, max_value=200.0, value=50.0)
    etime = st.number_input("Analysis Time (s)", min_value=0.1, max_value=3.0, value=2.0)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if ZAPIER_WEBHOOK_URL:
            # 5. [추가] Allrun에 전달할 파라미터 구성 강화
            payload = {
                "press": press_mpa * 1e6, # Pa로 변환
                "etime": etime,
                "nu": nu,
                "gate": {"x": gx, "y": gy, "z": gz},
                "mat": mat_name
            }
            # Zapier Webhook 전송 (에러 핸들링 추가)
            try:
                requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
                st.toast("✅ Action Triggered on GitHub Cloud!", icon="🌐")
                st.session_state["exec"] = True
            except requests.exceptions.RequestException as e:
                st.error(f"Zapier sync failed: {e}")

# --- 메인 화면: 3D 뷰어 + 조준점 시스템 ---
st.header("🎥 3D Geometry Analysis")

if uploaded_file:
    # 뷰어와 빨간 점을 감싸는 컨테이너 시작 (높이 고정)
    st.markdown('<div class="viewer-container">', unsafe_allow_html=True)
    st.markdown('<div class="red-dot"></div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-text">TARGET GATE POINT</div>', unsafe_allow_html=True)
    
    stl_from_file(
        file_path=os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl"), 
        color="#CCCCCC", # 조금 더 밝은 회색
        material="flat", # 질감 제거
        auto_rotate=False # 게이트 지정 시 흔들리지 않도록 자동 회전 끔
    )
    
    st.markdown('</div>', unsafe_allow_html=True) # 컨테이너 종료
    
    # [추가] 현재 지정된 좌표 요약 표시
    st.caption(f"📍 현재 조준점에 맞춘 게이트 좌표: X={gx}, Y={gy}, Z={gz}")
    st.info("모델을 마우스로 드래그하여 조준점에 맞추고 좌표를 입력하세요.")
else:
    st.info("왼쪽 사이드바에서 STL 파일을 업로드하세요.")

if st.session_state.get("exec"):
    st.success("🏃 GitHub Actions Solver is running with your gate position. Check 'Actions' tab.")
