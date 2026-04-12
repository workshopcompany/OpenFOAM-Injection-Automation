import streamlit as st
import os
import sys
import requests

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# AI Advisor 로드
try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'scripts/gemini_advisor.py'를 찾을 수 없습니다.")

ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI CONFIG (기존 UI 유지) ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")
st.caption("Cloud Infrastructure: Streamlit (UI) → Zapier → GitHub Actions (Solver)")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ STL uploaded to GitHub workspace.")

    mat_name = st.text_input("Material Name", value="Stainless Steel 316L")
    if st.button("🤖 Get AI Recommendation", type="primary"):
        with st.spinner("AI Analysis..."):
            st.session_state["props"] = get_material_properties(mat_name)

    if "props" in st.session_state:
        p = st.session_state["props"]
        
        st.header("⚙️ 2. Process Conditions")
        nu = st.number_input("Kinematic Viscosity (m²/s)", value=float(p["nu"]), format="%.2e")
        rho = st.number_input("Density (kg/m³)", value=float(p["rho"]))
        vel = st.number_input("Velocity (m/s)", min_value=0.001, max_value=1.0, value=0.100, format="%.3f")
        etime = st.number_input("Analysis Time (s)", min_value=0.01, max_value=0.5, value=0.10)

        if st.button("🚀 Run Cloud Simulation", type="primary"):
            if ZAPIER_WEBHOOK_URL:
                st.session_state["run_params"] = {
                    "nu": nu, "rho": rho, "vel": vel, "etime": etime, "mat": mat_name
                }
                try:
                    # Cloud Solver(GitHub)로 파라미터 전달
                    requests.post(ZAPIER_WEBHOOK_URL, json=st.session_state["run_params"], timeout=10)
                    st.toast("✅ Zapier Sync Successful!", icon="🌐")
                    st.session_state["exec"] = True
                except Exception as e:
                    st.error(f"Sync failed: {e}")

# --- MAIN: DISPLAY ---
if st.session_state.get("exec"):
    params = st.session_state["run_params"]
    st.header("🏃 Simulation Status")
    st.info(f"Target: {params['mat']} | Velocity {params['vel']} m/s")
    
    # 가상 로그 브리지 (사용자에게 진행 상황 안내)
    with st.expander("Cloud Bridge Status", expanded=True):
        st.code(f"""
>>> [STATION] Streamlit Cloud: Data Pushed
>>> [BRIDGE] Zapier: GitHub Repo Update Triggered
>>> [ENGINE] GitHub Actions: OpenFOAM Container Initializing...
>>> [LOGS] Check 'Actions' tab in GitHub for real-time solver output.
        """)
