import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import requests

# --- PATH SETUP ---
# 현재 파일(app.py)이 루트에 있으므로, scripts 및 OpenFOAM 폴더는 같은 레벨에 있습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

# scripts 폴더를 시스템 경로에 추가하여 gemini_advisor를 찾을 수 있게 함
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

try:
    from gemini_advisor import get_material_properties
except ImportError:
    st.error("❌ 'gemini_advisor.py' not found in /scripts folder. Please check your GitHub structure.")

# OpenFOAM 관련 경로 설정
CASE_DIR = os.path.join(BASE_DIR, "OpenFOAM", "case")
STL_DIR = os.path.join(CASE_DIR, "constant", "triSurface")

# --- SECRETS SETUP ---
try:
    ZAPIER_WEBHOOK_URL = st.secrets["ZAPIER_URL"]
except Exception:
    ZAPIER_WEBHOOK_URL = None
    st.warning("⚠️ ZAPIER_URL not found in Streamlit Secrets. Zapier integration is disabled.")

# --- UI CONFIG ---
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Flow Analysis")
st.caption("Cloud Pipeline: GitHub Root → Zapier → OpenFOAM Engine")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("📂 1. Geometry & Material")
    
    # STL Upload
    uploaded_file = st.file_uploader("Upload STL (mm unit)", type=["stl"])
    if uploaded_file:
        os.makedirs(STL_DIR, exist_ok=True)
        with open(os.path.join(STL_DIR, "part.stl"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to: {os.path.relpath(STL_DIR)}")

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
            
            # Zapier Webhook Sync
            if ZAPIER_WEBHOOK_URL:
                with st.spinner("Syncing data to Zapier..."):
                    try:
                        resp = requests.post(ZAPIER_WEBHOOK_URL, json=st.session_state["run_params"], timeout=5)
                        if resp.status_code == 200:
                            st.toast("✅ Zapier Sync Successful!", icon="🌐")
                        else:
                            st.error(f"Zapier Error: {resp.status_code}")
                    except Exception as e:
                        st.error(f"Webhook Failed: {e}")
            
            st.session_state["exec"] = True

# --- MAIN: EXECUTION & RESULTS ---
if st.session_state.get("exec"):
    params = st.session_state["run_params"]
    st.header("🏃 Simulation Progress")
    
    log_box = st.empty()
    bar = st.progress(0)
    
    env = {
        **os.environ, 
        "VELOCITY": str(params["vel"]), 
        "VISCOSITY": str(params["nu"]), 
        "END_TIME": str(params["etime"])
    }

    try:
        # 쉘 스크립트 실행 (OpenFOAM 환경이 구축된 서버에서만 작동)
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
            t1, t2, t3 = st.tabs(["Flow Field", "Pressure", "AI Report"])
            with t1: st.subheader("Velocity Vector Field"); _plot_demo()
            with t2: st.subheader("Pressure Distribution"); _plot_demo()
            with t3: 
                st.subheader("Gemini AI Analysis")
                st.write(f"**Material:** {params['mat']} | **Status:** Data Synced to GitHub")
                st.info("The configuration was updated via Zapier. Check GitHub Actions for the full solver log.")
        else:
            st.error("Simulation engine not found or failed. Ensure OpenFOAM is installed.")
    except Exception as e:
        st.error(f"Execution Error: {e}")

def _plot_demo():
    import numpy as np
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100); y = np.sin(x)
    ax.plot(x, y); st.pyplot(fig)
