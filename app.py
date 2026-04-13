import streamlit as st
import os
import sys
import requests

# --- PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'scripts/gemini_advisor.py' missing.")

ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

# --- UI ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: Cloud Simulation")

with st.sidebar:
    st.header("📂 1. Geometry & Material")
    uploaded_file = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded_file:
        stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ STL saved for GitHub Action.")

    st.subheader("🤖 AI Property Recommendation")
    mat_name = st.text_input("Material Name", value="PP")
    if st.button("Get AI Suggestion"):
        st.session_state["props"] = get_material_properties(mat_name)
    
    props = st.session_state.get("props", {"nu": 1e-6, "rho": 7800.0})

    st.header("⚙️ 2. Process Conditions")
    nu = st.number_input("Viscosity (m2/s)", value=float(props["nu"]), format="%.2e")
    vel = st.number_input("Velocity (m/s)", value=0.1, format="%.3f")
    etime = st.number_input("Analysis Time (s)", value=0.1)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if ZAPIER_WEBHOOK_URL:
            payload = {"vel": vel, "etime": etime, "nu": nu, "mat": mat_name}
            requests.post(ZAPIER_WEBHOOK_URL, json=payload)
            st.toast("✅ Action Triggered!")
            st.session_state["exec"] = True

if st.session_state.get("exec"):
    st.success("🏃 GitHub Actions Solver is running. Check your Repo 'Actions' tab.")
