import streamlit as st
import requests
import zipfile
import io
import os
import time
import json

# --- 1. 설정 및 보안 (Secrets 안전하게 불러오기) ---
def get_secret(key):
    try:
        return st.secrets[key]
    except KeyError:
        st.error(f"⚠️ 보안 설정(Secrets)에 **{key}**가 등록되지 않았습니다.")
        return None

# 설정된 이름에 맞춰 Secrets 로드
ZAPIER_WEBHOOK_URL = get_secret("ZAPIER_WEBHOOK_URL") 
GITHUB_TOKEN = get_secret("GITHUB_TOKEN")

REPO_OWNER = "workshopcompany"
REPO_NAME = "OpenFOAM-Injection-Automation"
ARTIFACT_NAME = "OpenFOAM-Web-Dashboard"

st.set_page_config(page_title="MIM-Ops Dashboard", layout="wide")

# 필수 설정이 없으면 중단
if not ZAPIER_WEBHOOK_URL or not GITHUB_TOKEN:
    st.warning("💡 Streamlit Cloud의 **Settings > Secrets** 설정을 확인해주세요.")
    st.stop()

# --- 2. 핵심 로직 함수 ---

def trigger_simulation(velocity, viscosity, end_time, label):
    """Zapier Webhook을 통해 GitHub Actions 실행 요청 (로그 포함)"""
    payload = {
        "velocity": str(velocity),
        "viscosity": str(viscosity),
        "end_time": str(end_time),
        "run_label": label
    }
    
    # UI에 전송 상태 로그 출력
    log_placeholder = st.empty()
    with log_placeholder.container():
        st.write("---")
        st.write("🔍 **통신 디버그 로그**")
        st.info(f"📤 Zapier로 보내는 데이터: `{json.dumps(payload)}`")
    
    try:
        res = requests.post(ZAPIER_WEBHOOK_URL, json=payload)
        
        with log_placeholder.container():
            st.write("---")
            st.write("🔍 **통신 디버그 로그**")
            st.info(f"📤 Zapier로 보내는 데이터: `{json.dumps(payload)}`")
            if res.status_code == 200:
                st.success(f"✅ Zapier 전송 성공! (Status: {res.status_code})")
                st.write("📥 Zapier 응답 메시지:")
                st.code(res.text)
            else:
                st.error(f"❌ Zapier 전송 실패 (Status: {res.status_code})")
                st.code(res.text)
        
        return res.status_code == 200
    except Exception as e:
        st.error(f"🚨 연결 오류: {e}")
        return False

def fetch_latest_artifact():
    """GitHub API를 사용하여 최신 시뮬레이션 결과 파일(zip) 가져오기"""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            artifacts = response.json().get("artifacts", [])
            target = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
            
            if target:
                download_url = target["archive_download_url"]
                file_res = requests.get(download_url, headers=headers)
                
                if file_res.status_code == 200:
                    result_dir = "temp_results"
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    
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

with st.sidebar:
    st.header("⚙️ 시뮬레이션 파라미터")
    vel = st.number_input("입구 유속 (Inlet Velocity)", value=2.5, step=0.1)
    vis = st.text_input("동점성 계수 (Kinematic Viscosity)", value="1e-6")
    etime = st.number_input("해석 시간 (End Time)", value=5, step=1)
    run_label = st.text_input("프로젝트 명", value=f"MIM_Project_{int(time.time())}")
    
    if st.button("🚀 시뮬레이션 시작", use_container_width=True):
        trigger_simulation(vel, vis, etime, run_label)

st.divider()
main_col, side_col = st.columns([2, 1])

with main_col:
    st.subheader("📊 시뮬레이션 분석 결과")
    if st.button("🔄 최신 결과 업데이트 (Refresh)", type="primary"):
        with st.spinner("GitHub에서 최신 데이터를 가져오는 중..."):
            success, result = fetch_latest_artifact()
            if success:
                st.session_state['data_path'] = result
            else:
                st.warning(result)

    if 'data_path' in st.session_state:
        res_dir = st.session_state['data_path']
        all_files = os.listdir(res_dir)
        
        imgs = sorted([f for f in all_files if f.endswith(".png")])
        if imgs:
            st.write("#### 🖼️ 유동 흐름 분석 이미지")
            grid = st.columns(3)
            for idx, img_file in enumerate(imgs):
                with grid[idx % 3]:
                    st.image(os.path.join(res_dir, img_file), caption=img_file, use_container_width=True)
        
        if "report.md" in all_files:
            with side_col:
                st.subheader("📄 분석 리포트")
                with open(os.path.join(res_dir, "report.md"), "r", encoding="utf-8") as f:
                    st.markdown(f.read())
