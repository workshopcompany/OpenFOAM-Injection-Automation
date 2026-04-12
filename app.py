import streamlit as st
import requests
import time
from datetime import datetime

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_OWNER = st.secrets["GITHUB_OWNER"]
GITHUB_REPO  = st.secrets["GITHUB_REPO"]
WORKFLOW_ID  = "run_openfoam.yml"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

st.set_page_config(page_title="MIM-Flow Analyzer", page_icon="🌊", layout="wide")
st.title("🌊 MIM-Flow Analyzer")
st.caption("Materials-Ops Portfolio | Auto-CFD Module v1.0")
st.divider()

with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    velocity = st.slider("Inlet Velocity (m/s)", 0.5, 10.0, 2.5, 0.5)
    viscosity_map = {
        "Water (1e-6)": "1e-6",
        "Oil (1e-4)":   "1e-4",
        "Air (1.5e-5)": "1.5e-5"
    }
    viscosity_label = st.selectbox("Fluid (Kinematic Viscosity)", list(viscosity_map.keys()))
    end_time  = st.slider("Simulation End Time (s)", 1, 20, 5, 1)
    run_label = st.text_input("Run Label", value=f"run-{datetime.now().strftime('%m%d-%H%M')}")
    st.divider()
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Current Parameters")
    st.json({"velocity": f"{velocity} m/s", "viscosity": viscosity_map[viscosity_label],
             "end_time": f"{end_time} s", "label": run_label})
with col2:
    st.subheader("📈 Reynolds Number Preview")
    Re = velocity * 0.05 / float(viscosity_map[viscosity_label])
    st.metric("Re", f"{Re:,.0f}")
    if Re < 2300:   st.success("🟢 Laminar Flow")
    elif Re < 4000: st.warning("🟡 Transitional Flow")
    else:           st.error("🔴 Turbulent Flow")

st.divider()

def trigger_workflow(velocity, viscosity, end_time, run_label):
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"
    res = requests.post(url, headers=HEADERS, json={
        "ref": "main",
        "inputs": {
            "velocity":  str(velocity),
            "viscosity": viscosity_map[viscosity],
            "end_time":  str(end_time),
            "run_label": run_label
        }
    })
    return res.status_code == 204

def get_run_after(triggered_at):
    """트리거 시점 이후에 생성된 run만 가져옴"""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs?per_page=5"
    res = requests.get(url, headers=HEADERS)
    for run in res.json().get("workflow_runs", []):
        if run["created_at"] >= triggered_at:
            return run
    return None

def get_artifacts_for_run(run_id):
    """특정 run_id의 artifact만 가져옴"""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}/artifacts"
    res = requests.get(url, headers=HEADERS)
    artifacts = res.json().get("artifacts", [])
    return artifacts[0] if artifacts else None

if run_button:
    triggered_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    with st.status("🔄 Simulation 실행 중...", expanded=True) as status:
        st.write("GitHub Actions에 파라미터 전송 중...")
        success = trigger_workflow(velocity, viscosity_label, end_time, run_label)

        if not success:
            status.update(label="❌ 워크플로우 트리거 실패", state="error")
            st.error("GITHUB_TOKEN 권한(repo + actions:write)을 확인하세요.")
            st.stop()

        st.write("✅ 워크플로우 트리거 완료! 실행 ID 확인 중...")
        time.sleep(8)  # GitHub API 반영 대기

        # 현재 실행 ID 확인
        current_run = None
        for _ in range(10):
            current_run = get_run_after(triggered_at)
            if current_run:
                break
            time.sleep(3)

        if not current_run:
            status.update(label="❌ 실행 ID를 찾을 수 없음", state="error")
            st.stop()

        run_id  = current_run["id"]
        run_url = current_run["html_url"]
        st.write(f"✅ Run ID: `{run_id}` | [GitHub에서 보기]({run_url})")

        # 완료 대기 (최대 15분)
        for i in range(90):
            res = requests.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}",
                headers=HEADERS
            )
            run_data   = res.json()
            run_status = run_data.get("status")
            run_conc   = run_data.get("conclusion")
            elapsed    = (i + 1) * 10

            st.write(f"  [{elapsed}s] status: {run_status} | conclusion: {run_conc}")

            if run_status == "completed":
                if run_conc == "success":
                    status.update(label="✅ 시뮬레이션 완료!", state="complete")
                else:
                    status.update(label=f"❌ 실패: {run_conc} | [로그 확인]({run_url})", state="error")
                break
            time.sleep(10)

        # 이 run의 artifact만 표시
        st.divider()
        st.subheader("🖼️ 시뮬레이션 결과")

        artifact = get_artifacts_for_run(run_id)
        if artifact:
            artifact_id = artifact["id"]
            st.success(f"Artifact ID: `{artifact_id}` — 생성 완료!")
            st.markdown(f"[📦 GitHub Actions에서 결과 다운로드]({run_url})")
        else:
            st.warning("Artifact가 아직 없습니다. 잠시 후 GitHub Actions 페이지에서 직접 확인하세요.")
            st.markdown(f"[🔗 이번 실행 결과 보기]({run_url})")
