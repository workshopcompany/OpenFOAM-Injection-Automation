import streamlit as st
import requests
import zipfile
import io
import os
import time

# --- 1. 설정 및 보안 ---
# Streamlit Cloud의 Settings > Secrets에 아래 값들을 설정하세요.
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
REPO_OWNER = "workshopcompany"
REPO_NAME = "OpenFOAM-Injection-Automation"
ARTIFACT_NAME = "OpenFOAM-Web-Dashboard"
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_WEB_URL", "") # 기존 Webhook URL

st.set_page_config(page_title="MIM-Ops Dashboard", layout="wide")

# --- 2. 핵심 로직 함수 ---

def trigger_simulation(velocity, viscosity, end_time, label):
    """Zapier Webhook을 통해 GitHub Actions 실행 요청"""
    payload = {
        "velocity": str(velocity),
        "viscosity": str(viscosity),
        "end_time": str(end_time),
        "run_label": label
    }
    try:
        res = requests.post(ZAPIER_WEBHOOK_URL, json=payload)
        return res.status_code == 200
    except Exception as e:
        st.error(f"연결 오류: {e}")
        return False

def fetch_latest_artifact():
    """GitHub API를 사용하여 최신 시뮬레이션 결과 파일(zip) 가져오기"""
    if not GITHUB_TOKEN:
        return False, "GitHub Token이 설정되지 않았습니다. Secrets를 확인하세요."
    
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            artifacts = response.json().get("artifacts", [])
            # 워크플로우에서 명시한 이름과 일치하는 가장 최신 Artifact 탐색
            target = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
            
            if target:
                download_url = target["archive_download_url"]
                file_res = requests.get(download_url, headers=headers)
                
                if file_res.status_code == 200:
                    result_dir = "temp_results"
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    
                    # zip 파일 압축 해제
                    with zipfile.ZipFile(io.BytesIO(file_res.content)) as zip_ref:
                        zip_ref.extractall(result_dir)
                    return True, result_dir
            return False, "아직 생성된 결과 파일이 없습니다. (약 2~3분 소요)"
        return False, f"GitHub 접근 실패 (코드: {response.status_code})"
    except Exception as e:
        return False, str(e)

# --- 3. UI 구성 ---

st.title("🚀 MIM-Ops: 유동해석 자동화 플랫폼")
st.info("금속 사출 성형(MIM) 공정 최적화를 위한 OpenFOAM 자동화 파이프라인입니다.")

# 사이드바: 입력 파라미터
with st.sidebar:
    st.header("⚙️ 시뮬레이션 파라미터")
    vel = st.number_input("입구 유속 (Inlet Velocity)", value=2.5, step=0.1)
    vis = st.text_input("동점성 계수 (Kinematic Viscosity)", value="1e-6")
    etime = st.number_input("해석 시간 (End Time)", value=5, step=1)
    run_label = st.text_input("프로젝트 명", value=f"MIM_Project_{int(time.time())}")
    
    if st.button("🚀 시뮬레이션 시작", use_container_width=True):
        with st.spinner("GitHub로 명령을 전송 중..."):
            if trigger_simulation(vel, vis, etime, run_label):
                st.success("해석이 시작되었습니다!")
                st.toast("GitHub Actions 실행 중...")
            else:
                st.error("트리거 실패. Webhook URL을 확인하세요.")

# 메인 화면: 결과 확인
st.divider()
main_col, side_col = st.columns([2, 1])

with main_col:
    st.subheader("📊 시뮬레이션 분석 결과")
    if st.button("🔄 최신 결과 업데이트 (Refresh)", type="primary"):
        with st.spinner("GitHub에서 최신 데이터를 가져오는 중..."):
            success, result = fetch_latest_artifact()
            if success:
                st.session_state['data_path'] = result
                st.success("데이터를 성공적으로 불러왔습니다.")
            else:
                st.warning(result)

    # 결과 데이터 렌더링
    if 'data_path' in st.session_state:
        res_dir = st.session_state['data_path']
        all_files = os.listdir(res_dir)
        
        # 이미지 파일 시각화
        imgs = sorted([f for f in all_files if f.endswith(".png")])
        if imgs:
            st.write("#### 🖼️ 유동 흐름 분석 이미지")
            grid = st.columns(3)
            for idx, img_file in enumerate(imgs):
                with grid[idx % 3]:
                    st.image(os.path.join(res_dir, img_file), caption=img_file, use_container_width=True)
        
        # 리포트 출력
        if "report.md" in all_files:
            with side_col:
                st.subheader("📄 분석 리포트")
                with open(os.path.join(res_dir, "report.md"), "r", encoding="utf-8") as f:
                    st.markdown(f.read())
