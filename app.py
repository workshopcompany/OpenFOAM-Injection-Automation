import streamlit as st
import requests
import json
import time
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────
GITHUB_TOKEN  = st.secrets["GITHUB_TOKEN"]
GITHUB_OWNER  = st.secrets["GITHUB_OWNER"]   # e.g. "workshopcompany"
GITHUB_REPO   = st.secrets["GITHUB_REPO"]    # e.g. "OpenFOAM-Injection-Automation"
WORKFLOW_ID   = "run_openfoam.yml"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

# ── 페이지 설정 ────────────────────────────────────────
st.set_page_config(
    page_title="MIM-Flow Analyzer",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 MIM-Flow Analyzer")
st.caption("Materials-Ops Portfolio | Auto-CFD Module v1.0")
st.divider()

# ── 사이드바: 입력 파라미터 ────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Parameters")

    velocity = st.slider(
        "Inlet Velocity (m/s)",
        min_value=0.5, max_value=10.0,
        value=2.5, step=0.5
    )
    viscosity = st.selectbox(
        "Fluid (Kinematic Viscosity)",
        options={
            "Water (1e-6)": "1e-6",
            "Oil (1e-4)":   "1e-4",
            "Air (1.5e-5)": "1.5e-5"
        }.keys()
    )
    viscosity_map = {
        "Water (1e-6)": "1e-6",
        "Oil (1e-4)":   "1e-4",
        "Air (1.5e-5)": "1.5e-5"
    }
    end_time = st.slider(
        "Simulation End Time (s)",
        min_value=1, max_value=20,
        value=5, step=1
    )
    run_label = st.text_input(
        "Run Label",
        value=f"run-{datetime.now().strftime('%m%d-%H%M')}"
    )

    st.divider()
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ── 메인: 상태 및 결과 ─────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Current Parameters")
    st.json({
        "velocity":  f"{velocity} m/s",
        "viscosity": viscosity_map[viscosity],
        "end_time":  f"{end_time} s",
        "label":     run_label
    })

with col2:
    st.subheader("📈 Reynolds Number Preview")
    L = 0.05  # characteristic length (m) — 채널 높이
    nu = float(viscosity_map[viscosity])
    Re = velocity * L / nu
    st.metric("Re", f"{Re:,.0f}")
    if Re < 2300:
        st.success("🟢 Laminar Flow")
    elif Re < 4000:
        st.warning("🟡 Transitional Flow")
    else:
        st.error("🔴 Turbulent Flow")

st.divider()

# ── 시뮬레이션 트리거 ──────────────────────────────────
def trigger_workflow(velocity, viscosity, end_time, run_label):
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"
    payload = {
        "ref": "main",
        "inputs": {
            "velocity":   str(velocity),
            "viscosity":  viscosity_map[viscosity],
            "end_time":   str(end_time),
            "run_label":  run_label
        }
    }
    res = requests.post(url, headers=HEADERS, json=payload)
    return res.status_code == 204

def get_latest_run():
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs?per_page=1"
    res = requests.get(url, headers=HEADERS)
    runs = res.json().get("workflow_runs", [])
    return runs[0] if runs else None

def get_latest_artifact_url():
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/artifacts?per_page=1"
    res = requests.get(url, headers=HEADERS)
    artifacts = res.json().get("artifacts", [])
    if artifacts:
        return artifacts[0]["archive_download_url"], artifacts[0]["id"]
    return None, None

if run_button:
    with st.status("🔄 Submitting simulation...", expanded=True) as status:
        st.write("Sending parameters to GitHub Actions...")
        success = trigger_workflow(velocity, viscosity, end_time, run_label)

        if not success:
            status.update(label="❌ Failed to trigger workflow", state="error")
            st.error("GitHub API 호출 실패. GITHUB_TOKEN 권한을 확인하세요.")
            st.stop()

        st.write("✅ Workflow triggered! Waiting for completion...")
        time.sleep(5)

        # 완료 대기 (최대 10분)
        for i in range(60):
            run = get_latest_run()
            if run:
                run_status     = run.get("status")
                run_conclusion = run.get("conclusion")
                st.write(f"  [{i*10}s] status: {run_status} | conclusion: {run_conclusion}")

                if run_status == "completed":
                    if run_conclusion == "success":
                        status.update(label="✅ Simulation complete!", state="complete")
                    else:
                        status.update(label=f"❌ Simulation failed: {run_conclusion}", state="error")
                    break
            time.sleep(10)

# ── 결과 표시 ──────────────────────────────────────────
st.subheader("🖼️ Latest Simulation Results")

artifact_url, artifact_id = get_latest_artifact_url()

if artifact_id:
    st.info(f"Artifact ID: `{artifact_id}` — GitHub Actions에서 다운로드 가능")
    st.markdown(
        f"[📦 Download Results](https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions)",
        unsafe_allow_html=False
    )
else:
    st.warning("아직 실행된 결과가 없습니다. 시뮬레이션을 실행해보세요.")

st.divider()
st.caption("MIM-Flow Analyzer · Materials-Ops Portfolio · Auto-CFD Module v1.0")
