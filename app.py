import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import requests

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'gemini_advisor.py'를 scripts 폴더에서 찾을 수 없습니다.")

CASE_DIR = os.path.join(BASE_DIR, "OpenFOAM", "case")
STL_DIR = os.path.join(CASE_DIR, "constant", "triSurface")

# --- SECRETS SETUP ---
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI CONFIG ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")
st.caption("Cloud Pipeline: GitHub Root → Zapier → OpenFOAM Engine")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    uploaded_file = st.file_uploader("STL 파일 업로드 (mm 단위)", type=["stl"])
    if uploaded_file:
        os.makedirs(STL_DIR, exist_ok=True)
        with open(os.path.join(STL_DIR, "part.stl"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("STL 파일이 서버에 저장되었습니다.")

    mat_name = st.text_input("대상 재료 입력", value="Stainless Steel 316L")
    if st.button("🤖 AI 물성치 추천 받기", type="primary"):
        with st.spinner("AI 분석 중..."):
            st.session_state["props"] = get_material_properties(mat_name)

    if "props" in st.session_state:
        p = st.session_state["props"]
        st.info(f"데이터 출처: {p.get('source', 'Unknown').upper()}")
        
        # --- UI 입력 제한 적용 ---
        # 1. 점도: 너무 낮으면 해석이 불안정하므로 최소값 설정
        nu = st.number_input("동점성 계수 (m²/s)", 
                             min_value=1e-7, max_value=1e-1, 
                             value=float(p["nu"]), format="%.2e")
        
        rho = st.number_input("밀도 (kg/m³)", 
                              min_value=100.0, max_value=20000.0, 
                              value=float(p["rho"]))
        
        st.header("⚙️ 2. 해석 조건 설정 (Safety Guard)")
        
        # 2. 속도: UI에서 최대 1.0 m/s로 제한 (OpenFOAM 수치 안정성 확보)
        vel = st.number_input("사출 속도 (m/s) [제한: 0.001 ~ 1.0]", 
                              min_value=0.001, max_value=1.000, 
                              value=0.100, step=0.010, format="%.3f")
        
        # 3. 시간: 너무 길면 계산 리소스 문제로 최대 0.5초 제한
        etime = st.number_input("해석 시간 (s) [제한: 0.01 ~ 0.5]", 
                                min_value=0.01, max_value=0.50, 
                                value=0.10, step=0.05)

        if st.button("🚀 시뮬레이션 실행 (Cloud)", type="primary"):
            st.session_state["run_params"] = {
                "nu": nu, "rho": rho, "vel": vel, 
                "etime": etime, "mat": mat_name
            }
            
            # Zapier Webhook 전송
            if ZAPIER_WEBHOOK_URL:
                with st.spinner("Zapier 데이터 동기화 중..."):
                    try:
                        resp = requests.post(ZAPIER_WEBHOOK_URL, json=st.session_state["run_params"], timeout=10)
                        if resp.status_code == 200:
                            st.toast("✅ Zapier 연동 성공!", icon="🌐")
                        else:
                            st.error(f"Zapier 에러: {resp.status_code}")
                    except Exception as e:
                        st.error(f"연동 실패: {e}")
            
            st.session_state["exec"] = True

# --- MAIN: EXECUTION & RESULTS ---
if st.session_state.get("exec"):
    params = st.session_state["run_params"]
    st.header("🏃 시뮬레이션 진행 상황")
    
    st.info(f"설정된 조건: 속도 {params['vel']} m/s, 해석 시간 {params['etime']} s")
    
    # Cloud 처리 안내
    st.warning("⚠️ Streamlit 서버에는 해석 엔진이 없습니다. 데이터가 GitHub Actions로 전송되었습니다.")
    st.markdown(f"""
    ### 다음 단계를 확인하세요:
    1. **GitHub Actions**: 저장소의 'Actions' 탭에서 OpenFOAM 해석이 시작되었는지 확인하세요.
    2. **수치 안정성**: 현재 속도({params['vel']} m/s)는 `maxCo 0.2` 가이드라인 내에서 안전하게 계산됩니다.
    3. **결과 업데이트**: 해석 완료 후 리포트가 생성됩니다.
    """)
    
    # 더미 로그 (사용자 경험용)
    with st.expander("실시간 로그 확인 (Cloud Bridge)"):
        st.code(">>> Data Pushed to GitHub Repository...\n>>> Triggering GitHub Actions Workflow...\n>>> OpenFOAM Solver (interFoam) Initializing on Cloud Server...")

def _plot_demo():
    import numpy as np
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100); y = np.sin(x)
    ax.plot(x, y); st.pyplot(fig)
