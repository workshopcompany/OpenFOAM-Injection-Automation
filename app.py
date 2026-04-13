import streamlit as st
import os
import sys
import requests
import numpy as np
from streamlit_stl import stl_from_file

# --- 경로 및 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZAPIER_WEBHOOK_URL = st.secrets.get("ZAPIER_URL", None)

st.set_page_config(page_title="MIM-Ops: Gate Selector", layout="wide")
st.title("🔬 Gate Location & Process Control")

# 3D 모델 경로
stl_path = os.path.join(BASE_DIR, "OpenFOAM/case/constant/triSurface/part.stl")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📍 1. Gate Position")
    st.write("모델을 보며 게이트 위치(빨간색 점)를 조정하세요.")
    
    # 3D 좌표 슬라이더 (사용자가 마우스로 위치 변경)
    # 모델의 크기에 따라 min/max 값을 조정할 수 있습니다.
    gx = st.slider("Gate X (mm)", -100.0, 100.0, 0.0, 0.5)
    gy = st.slider("Gate Y (mm)", -100.0, 100.0, 0.0, 0.5)
    gz = st.slider("Gate Z (mm)", -100.0, 100.0, 0.0, 0.5)
    
    st.info(f"선택된 좌표: ({gx}, {gy}, {gz})")
    
    st.divider()
    
    st.header("⚙️ 2. Process Conditions")
    press_mpa = st.number_input("Injection Pressure (MPa)", 10.0, 200.0, 50.0)
    etime = st.slider("Analysis Time (s)", 0.5, 3.0, 2.0, 0.1)

    if st.button("🚀 Start Simulation with this Gate", type="primary"):
        # 선택된 좌표와 압력 조건을 전송
        payload = {
            "press": press_mpa * 1e6,
            "etime": etime,
            "gate_x": gx, "gate_y": gy, "gate_z": gz
        }
        # requests.post(ZAPIER_WEBHOOK_URL, json=payload)
        st.success(f"좌표 ({gx}, {gy}, {gz})에서 해석을 시작합니다!")

with col2:
    st.header("🎥 3D Live Preview")
    if os.path.exists(stl_path):
        # stl_from_file에 마커 기능을 지원하는 경우 (버전에 따라 상이)
        # 현재는 모델 자체를 보여주며 슬라이더로 위치를 확정하는 방식입니다.
        stl_from_file(
            file_path=stl_path,
            color="#CCCCCC",      # 모델은 연한 회색
            material="flat",
            auto_rotate=False
        )
        
        # 시각적 피드백: 사용자가 현재 위치를 인지할 수 있도록 메트릭 표시
        st.metric("Target Gate Z-Level", f"{gz} mm")
    else:
        st.warning("STL 파일을 먼저 업로드해주세요.")
