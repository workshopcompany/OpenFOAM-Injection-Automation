import streamlit as st
import os
import sys
import requests

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# AI Advisor 모듈 로드
try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'scripts/gemini_advisor.py'를 찾을 수 없습니다. 경로를 확인해주세요.")

# Zapier Webhook URL
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI CONFIG ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    # STL 업로드 로직
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ STL saved to Cloud storage.")

    # AI 물성치 추천 섹션 (복구됨)
    st.subheader("🤖 AI Material Advisor")
    mat_name = st.text_input("Material Name", value="Stainless Steel 316L")
    
    if st.button("Get AI Recommendation", type="secondary"):
        with st.spinner("AI analyzing material properties..."):
            # gemini_advisor.py의 함수 호출
            st.session_state["props"] = get_material_properties(mat_name)
            st.success(f"Recommended properties for {mat_name} loaded!")

    # AI 추천값 또는 사용자 입력값 설정
    st.divider()
    st.header("⚙️ 2. Process Conditions")

    # AI 추천값이 있으면 기본값으로 사용, 없으면 표준값 사용
    props = st.session_state.get("props", {"nu": 1e-6, "rho": 7800.0})
    
    nu = st.number_input("Kinematic Viscosity (m²/s)", 
                         min_value=1e-7, max_value=1e-1, 
                         value=float(props["nu"]), format="%.2e")
    
    rho = st.number_input("Density (kg/m³)", 
                          min_value=100.0, max_value=20000.0, 
                          value=float(props["rho"]))
    
    vel = st.number_input("Injection Velocity (m/s)", 
                          min_value=0.001, max_value=1.0, 
                          value=0.100, step=0.010, format="%.3f")
    
    etime = st.number_input("Analysis Time (s)", 
                            min_value=0.01, max_value=0.5, 
                            value=0.10, step=0.05)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if ZAPIER_WEBHOOK_URL:
            payload = {
                "vel": vel, 
                "etime": etime, 
                "nu": nu, 
                "rho": rho,
                "mat": mat_name
            }
            try:
                # Zapier를 통해 GitHub Actions 트리거
                requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
                st.toast("✅ Zapier Sync Successful!", icon="🌐")
                st.session_state["exec"] = True
            except Exception as e:
                st.error(f"Sync failed: {e}")
        else:
            st.error("⚠️ ZAPIER_URL이 설정되지 않았습니다.")

# --- MAIN: DISPLAY ---
if st.session_state.get("exec"):
    st.success("🏃 Simulation Pipeline Active!")
    st.info(f"Target: {mat_name} | Velocity: {vel} m/s | Time: {etime} s")
    
    st.markdown("""
    ### 🛠️ Next Steps:
    1. **GitHub Actions**: 저장소의 'Actions' 탭에서 시뮬레이션 로그를 실시간으로 확인하세요.
    2. **Log System**: 각 단계별(Mesh, Solver, Scale) 로그가 독립적으로 생성됩니다.
    3. **Results**: 해석 완료 후 상단 Artifacts에서 결과를 다운로드할 수 있습니다.
    """)
    
    with st.expander("Cloud Bridge Status"):
        st.code(">>> Environment: OpenFOAM v2312 (opencfd/openfoam-dev)\n"
                ">>> Data Sync: Completed\n"
                ">>> Solver: interFoam initializing with maxCo 0.2 Guard...")
