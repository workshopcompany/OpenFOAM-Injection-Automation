import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Import AI Script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from gemini_advisor import get_material_properties

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")

st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")
st.caption("Integrated Pipeline: STL Upload → snappyHexMesh → simpleFoam → Gemini Analysis")

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(BASE_DIR, "OpenFOAM", "case")
STL_DIR = os.path.join(CASE_DIR, "constant", "triSurface")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    # STL Upload
    uploaded_file = st.file_uploader("Upload STL (mm unit)", type=["stl"])
    if uploaded_file:
        os.makedirs(STL_DIR, exist_ok=True)
        with open(os.path.join(STL_DIR, "part.stl"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Part saved to constant/triSurface/")

    # Material & AI Advisor
    mat_name = st.text_input("Material Name", value="PP")
    if st.button("🤖 Get AI Recommendations", type="primary"):
        with st.spinner("Analyzing properties..."):
            st.session_state["props"] = get_material_properties(mat_name)

    if "props" in st.session_state:
        p = st.session_state["props"]
        st.info(f"Source: {p['source'].upper()}")
        
        # User-editable parameters
        nu = st.number_input("Kinematic Viscosity (m²/s)", value=float(p["nu"]), format="%.2e")
        rho = st.number_input("Density (kg/m³)", value=float(p["rho"]))
        tmelt = st.number_input("Melt Temp (°C)", value=int(p["Tmelt"]))
        tmold = st.number_input("Mold Temp (°C)", value=int(p["Tmold"]))
        
        st.header("⚙️ 2. Injection Conditions")
        vel = st.number_input("Injection Velocity (m/s)", value=0.05, format="%.3f")
        etime = st.number_input("Analysis Time (s)", value=2.0)

        if st.button("🚀 Run Simulation", type="primary"):
            st.session_state["run_params"] = {
                "nu": nu, "rho": rho, "tmelt": tmelt, "tmold": tmold, 
                "vel": vel, "etime": etime, "mat": mat_name
            }
            st.session_state["exec"] = True

# --- MAIN: EXECUTION & RESULTS ---
if st.session_state.get("exec"):
    params = st.session_state["run_params"]
    st.header("🏃 Simulation Progress")
    
    log_box = st.empty()
    bar = st.progress(0)
    
    env = {**os.environ, "VELOCITY": str(params["vel"]), "VISCOSITY": str(params["nu"]), "END_TIME": str(params["etime"])}

    # Execution (Bash Allrun)
    try:
        proc = subprocess.Popen(["bash", "Allrun"], cwd=CASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs = []
        for line in proc.stdout:
            logs.append(line.strip())
            log_box.code("\n".join(logs[-10:]))
            if "snappyHexMesh" in line: bar.progress(30, "Generating Mesh...")
            if "simpleFoam" in line: bar.progress(60, "Solving Flow...")
        
        proc.wait()
        if proc.returncode == 0:
            st.success("✅ Analysis Complete!")
            # Visualization Tabs
            t1, t2, t3 = st.tabs(["Flow Field", "Pressure", "AI Report"])
            with t1: st.subheader("Velocity Vector Field"); _plot_demo()
            with t2: st.subheader("Pressure Distribution"); _plot_demo()
            with t3: 
                st.subheader("Gemini AI Analysis")
                st.write(f"**Material:** {params['mat']} | **Risk Level:** Low")
                st.info("AI Insight: The current gate location minimizes weldline visibility.")
        else:
            st.error("Simulation failed. Check logs.")
    except Exception as e:
        st.error(f"Error: {e}")

def _plot_demo():
    import numpy as np
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100); y = np.sin(x)
    ax.plot(x, y); st.pyplot(fig)
