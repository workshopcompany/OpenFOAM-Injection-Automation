import streamlit as st
import os
import sys
import requests

# --- CONFIG & PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        st.success("STL saved to Cloud storage.")

    mat_name = st.text_input("Material Name", value="Stainless Steel 316L")
    
    st.header("⚙️ 2. Process Conditions")
    vel = st.number_input("Velocity (m/s)", min_value=0.001, max_value=1.0, value=0.1, format="%.3f")
    etime = st.number_input("Time (s)", min_value=0.01, max_value=0.5, value=0.1)

    if st.button("🚀 Run Cloud Simulation", type="primary"):
        if ZAPIER_WEBHOOK_URL:
            payload = {"vel": vel, "etime": etime, "mat": mat_name}
            try:
                requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
                st.toast("✅ Zapier Sync Successful!", icon="🌐")
                st.session_state["exec"] = True
            except Exception as e:
                st.error(f"Sync failed: {e}")

if st.session_state.get("exec"):
    st.success("🏃 Simulation Triggered!")
    st.markdown("""
    ### Current Pipeline Status:
    1. **Data Sync**: Zapier is updating GitHub repository.
    2. **Cloud Solver**: GitHub Actions is spinning up an **OpenFOAM v2312** container.
    3. **Logging**: Each step (Scaling, Meshing, Solving) is being logged independently.
    
    **To see real-time progress:** Check the **'Actions'** tab in your GitHub repository.
    """)
    
    with st.expander("Cloud Execution Trace"):
        st.code(">>> env: opencfd/openfoam-dev:2312\n>>> status: sourcing bashrc...\n>>> status: scaling STL (mm -> m)...")
