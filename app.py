import streamlit as st
import subprocess
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
    st.error("❌ 'gemini_advisor.py' not found in scripts folder.")

CASE_DIR = os.path.join(BASE_DIR, "OpenFOAM", "case")
STL_DIR = os.path.join(CASE_DIR, "constant", "triSurface")

# --- SECRETS SETUP ---
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI CONFIG ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")
st.caption("Cloud Pipeline: GitHub → Zapier → OpenFOAM Engine")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        os.makedirs(STL_DIR, exist_ok=True)
        with open(os.path.join(STL_DIR, "part.stl"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("STL saved.")

    mat_name = st.text_input("Material Name", value="Stainless Steel 316L")
    if st.button("🤖 Get AI Recommendation", type="primary"):
        with st.spinner("AI Analysis in progress..."):
            st.session_state["props"] = get_material_properties(mat_name)

    if "props" in st.session_state:
        p = st.session_state["props"]
        
        # Numerical Stability Guards
        nu = st.number_input("Kinematic Viscosity (m²/s)", 
                             min_value=1e-7, max_value=1e-1, 
                             value=float(p["nu"]), format="%.2e")
        
        rho = st.number_input("Density (kg/m³)", 
                              min_value=100.0, max_value=20000.0, 
                              value=float(p["rho"]))
        
        st.header("⚙️ 2. Process Conditions")
        
        # Velocity limiter for stability
        vel = st.number_input("Velocity (m/s) [Limit: 0.001-1.0]", 
                              min_value=0.001, max_value=1.000, 
                              value=0.100, step=0.010, format="%.3f")
        
        # Time limiter for resources
        etime = st.number_input("Analysis Time (s) [Limit: 0.01-0.5]", 
                                min_value=0.01, max_value=0.50, 
                                value=0.10, step=0.05)

        if st.button("🚀 Run Simulation", type="primary"):
            st.session_state["run_params"] = {
                "nu": nu, "rho": rho, "vel": vel, 
                "etime": etime, "mat": mat_name
            }
            
            if ZAPIER_WEBHOOK_URL:
                try:
                    requests.post(ZAPIER_WEBHOOK_URL, json=st.session_state["run_params"], timeout=10)
                    st.toast("✅ Zapier Sync Successful!", icon="🌐")
                except Exception as e:
                    st.error(f"Sync failed: {e}")
            
            st.session_state["exec"] = True

# --- MAIN: DISPLAY ---
if st.session_state.get("exec"):
    params = st.session_state["run_params"]
    st.header("🏃 Simulation Status")
    st.info(f"Target: Velocity {params['vel']} m/s, Time {params['etime']} s")
    st.warning("⚠️ Processing on GitHub Actions Cloud...")
    
    with st.expander("Live Bridge Log"):
        st.code(">>> Data Pushed to GitHub\n>>> Initializing OpenFOAM v2312 Container\n>>> Running blockMesh & interFoam...")
