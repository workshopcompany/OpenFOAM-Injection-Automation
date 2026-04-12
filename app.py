import streamlit as st
import os
import sys
import requests

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'gemini_advisor.py' not found.")

ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")

with st.sidebar:
    st.header("📂 1. Geometry & Material")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("STL saved.")

    mat_name = st.text_input("Material Name", value="Stainless Steel 316L")
    if st.button("🤖 Get AI Recommendation", type="primary"):
        st.session_state["props"] = get_material_properties(mat_name)

    if "props" in st.session_state:
        p = st.session_state["props"]
        nu = st.number_input("Viscosity (m²/s)", min_value=1e-7, max_value=1e-1, value=float(p["nu"]), format="%.2e")
        rho = st.number_input("Density (kg/m³)", value=float(p["rho"]))
        
        st.header("⚙️ 2. Process Conditions")
        vel = st.number_input("Velocity (m/s)", min_value=0.001, max_value=1.0, value=0.1, format="%.3f")
        etime = st.number_input("Time (s)", min_value=0.01, max_value=0.5, value=0.1)

        if st.button("🚀 Run Cloud Simulation", type="primary"):
            if ZAPIER_WEBHOOK_URL:
                run_params = {"nu": nu, "rho": rho, "vel": vel, "etime": etime, "mat": mat_name}
                try:
                    requests.post(ZAPIER_WEBHOOK_URL, json=run_params, timeout=10)
                    st.toast("✅ Zapier Sync Successful!", icon="🌐")
                    st.session_state["exec"] = True
                except Exception as e:
                    st.error(f"Sync failed: {e}")

if st.session_state.get("exec"):
    st.info("🏃 Simulation is processing on GitHub Actions. Please check your GitHub Repository tab.")
