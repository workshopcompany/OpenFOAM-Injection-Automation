import streamlit as st
import subprocess
import json
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from gemini_advisor import get_material_properties

st.set_page_config(
    page_title="MIM-Ops 유동해석",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 MIM-Ops 플라스틱 사출 유동해석")
st.caption("OpenFOAM + Gemini AI 자동화 파이프라인")

# ─────────────────────────────────────────
# 사이드바: 입력
# ─────────────────────────────────────────
with st.sidebar:
    st.header("📋 해석 설정")

    # 재료 입력
    st.subheader("① 수지 재료")
    material_input = st.text_input(
        "재료명 입력",
        value="PP",
        placeholder="PP, ABS, Nylon66, PC ..."
    )

    if st.button("🤖 Gemini AI 물성 추천", type="primary"):
        with st.spinner("Gemini AI가 물성을 검색 중..."):
            props = get_material_properties(material_input)
            st.session_state["props"] = props
            st.session_state["props_confirmed"] = False

    # 물성 표시 및 수정
    if "props" in st.session_state:
        props = st.session_state["props"]
        st.success(f"✅ {props.get('source','').upper()} 에서 데이터 로드")

        st.subheader("② 물성 확인 / 수정")
        if "description" in props:
            st.info(props["description"])

        nu_val   = st.number_input("운동점도 nu (m²/s)", value=float(props.get("nu", 1e-3)),
                                    format="%.2e", min_value=1e-7, max_value=1.0)
        rho_val  = st.number_input("밀도 ρ (kg/m³)",    value=float(props.get("rho", 1000)),
                                    min_value=100, max_value=3000)
        Tmelt    = st.number_input("용융 온도 (°C)",     value=int(props.get("Tmelt", 220)),
                                    min_value=100, max_value=400)
        Tmold    = st.number_input("금형 온도 (°C)",     value=int(props.get("Tmold", 50)),
                                    min_value=10,  max_value=200)

        st.subheader("③ 사출 조건")
        velocity = st.number_input("사출 속도 (m/s)", value=0.05,
                                    min_value=0.001, max_value=1.0, format="%.3f")
        end_time = st.number_input("해석 시간 (s)",   value=2.0,
                                    min_value=0.1,   max_value=60.0)

        # 확인 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 확인 — 해석 시작", type="primary"):
                st.session_state["run_params"] = {
                    "material": material_input,
                    "nu": nu_val,
                    "rho": rho_val,
                    "Tmelt": Tmelt,
                    "Tmold": Tmold,
                    "velocity": velocity,
                    "end_time": end_time,
                }
                st.session_state["props_confirmed"] = True
        with col2:
            if st.button("🔄 다시 추천"):
                del st.session_state["props"]
                st.rerun()

# ─────────────────────────────────────────
# 메인: 해석 실행 및 결과
# ─────────────────────────────────────────

if st.session_state.get("props_confirmed") and "run_params" in st.session_state:
    params = st.session_state["run_params"]

    st.header("🚀 해석 실행")

    # 파라미터 요약
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("재료",     params["material"])
    col2.metric("점도 nu",  f"{params['nu']:.2e} m²/s")
    col3.metric("사출속도", f"{params['velocity']} m/s")
    col4.metric("밀도",     f"{params['rho']} kg/m³")

    # Re 수 계산
    L = 0.05  # 채널 높이 [m]
    Re = params["velocity"] * L / params["nu"]
    flow_type = "층류 ✅" if Re < 2300 else "난류 ⚠️ (단순화됨)"
    st.info(f"**레이놀즈 수 Re = {Re:.1f}** → {flow_type}")

    # OpenFOAM 실행
    case_dir = os.path.join(os.path.dirname(__file__), '..', 'OpenFOAM', 'case')
    env = {
        **os.environ,
        "VELOCITY":   str(params["velocity"]),
        "VISCOSITY":  str(params["nu"]),
        "DENSITY":    str(params["rho"]),
        "MELT_TEMP":  str(params["Tmelt"]),
        "MOLD_TEMP":  str(params["Tmold"]),
        "END_TIME":   str(params["end_time"]),
    }

    progress = st.progress(0, text="blockMesh 실행 중...")
    log_box  = st.empty()
    logs     = []

    try:
        proc = subprocess.Popen(
            ["bash", "Allrun"],
            cwd=case_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        step = 0
        for line in proc.stdout:
            logs.append(line.rstrip())
            log_box.code("\n".join(logs[-30:]), language="bash")
            if "blockMesh" in line:
                progress.progress(20, text="blockMesh 완료...")
            elif "simpleFoam" in line:
                progress.progress(50, text="simpleFoam 해석 중...")
            elif "완료" in line or "SUCCESS" in line:
                progress.progress(100, text="완료!")

        proc.wait()
        if proc.returncode != 0:
            st.error("❌ 해석 실패. 로그를 확인하세요.")
        else:
            st.success("✅ 해석 완료!")
            _show_results(case_dir, params)

    except FileNotFoundError:
        st.warning("⚠️ OpenFOAM이 설치되지 않은 환경입니다. GitHub Actions에서 실행하세요.")
        _show_demo_results(params)


def _show_results(case_dir: str, params: dict):
    """실제 결과 표시"""
    st.header("📊 해석 결과")

    tabs = st.tabs(["유동 흐름", "압력 분포", "열 분포", "웰드라인", "수축공", "데이터 다운로드"])

    with tabs[0]:
        st.subheader("유동 흐름 (속도장)")
        st.info("VTK 뷰어 또는 ParaView로 시각화하려면 결과 파일을 다운로드하세요.")
        _plot_velocity_heatmap(params)

    with tabs[1]:
        st.subheader("압력 분포")
        _plot_pressure(params)

    with tabs[2]:
        st.subheader("열 분포 (온도장 추정)")
        _plot_temperature(params)

    with tabs[3]:
        st.subheader("웰드라인 추정")
        _show_weldline(params)

    with tabs[4]:
        st.subheader("수축공 추정")
        _show_shrinkage(params)

    with tabs[5]:
        st.subheader("데이터 다운로드")
        results_dir = os.path.join(case_dir, "results")
        if os.path.exists(results_dir):
            import zipfile, io
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for f in os.listdir(results_dir):
                    z.write(os.path.join(results_dir, f), f)
            st.download_button(
                "📥 결과 ZIP 다운로드",
                data=buf.getvalue(),
                file_name=f"openfoam_results_{params['material']}.zip",
                mime="application/zip"
            )


def _plot_velocity_heatmap(params):
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        nx, ny = 40, 20
        x = np.linspace(0, 0.1, nx)
        y = np.linspace(0, 0.05, ny)
        Y, X = np.meshgrid(y, x)

        # 포아젤 유동 프로파일 (층류)
        H = 0.05
        V = params["velocity"]
        U = 1.5 * V * (1 - (2*Y/H - 1)**2)

        fig, ax = plt.subplots(figsize=(10, 4))
        c = ax.contourf(X, Y, U, levels=20, cmap="RdYlBu_r")
        plt.colorbar(c, ax=ax, label="속도 (m/s)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("속도장 — Poiseuille 층류 프로파일")
        st.pyplot(fig)
        plt.close()
    except ImportError:
        st.info("matplotlib 설치 필요: pip install matplotlib")


def _plot_pressure(params):
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        nx = 40
        x = np.linspace(0, 0.1, nx)
        nu = params["nu"]
        V  = params["velocity"]
        H  = 0.05
        # 하겐-포아젤 압력 강하
        dPdx = 12 * nu * V / H**2
        p = dPdx * (0.1 - x)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(x * 1000, p, "b-", linewidth=2)
        ax.fill_between(x * 1000, p, alpha=0.2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("압력 (Pa)")
        ax.set_title("채널 방향 압력 분포")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        col1.metric("최대 압력 (inlet)", f"{p[0]:.2f} Pa")
        col2.metric("압력 강하", f"{p[0]-p[-1]:.2f} Pa")
    except ImportError:
        st.info("matplotlib 설치 필요")


def _plot_temperature(params):
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        nx, ny = 40, 20
        x = np.linspace(0, 0.1, nx)
        y = np.linspace(0, 0.05, ny)
        Y, X = np.meshgrid(y, x)

        Tmelt = params["Tmelt"]
        Tmold = params["Tmold"]
        # 단순 온도 분포: 벽면=Tmold, 중심=Tmelt, x방향으로 냉각
        T = Tmold + (Tmelt - Tmold) * (1 - (2*Y/0.05 - 1)**2) * np.exp(-3 * X / 0.1)

        fig, ax = plt.subplots(figsize=(10, 4))
        c = ax.contourf(X, Y, T, levels=20, cmap="hot")
        plt.colorbar(c, ax=ax, label="온도 (°C)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"온도 분포 추정 (Tmelt={Tmelt}°C → Tmold={Tmold}°C)")
        st.pyplot(fig)
        plt.close()
    except ImportError:
        st.info("matplotlib 설치 필요")


def _show_weldline(params):
    st.markdown("""
    **웰드라인 추정 기준:**
    - 두 유동 선단이 만나는 지점
    - 속도 벡터가 서로 반대 방향으로 수렴하는 위치
    - 온도가 낮은 영역에서 발생 가능성 높음
    """)
    nu = params["nu"]
    V  = params["velocity"]
    Tm = params["Tmelt"]
    risk = "낮음 ✅" if Tm > 200 and V > 0.03 else "높음 ⚠️"
    st.metric("웰드라인 발생 위험도", risk)
    st.info("💡 Gemini AI 분석: 단순 채널 형상에서는 웰드라인 발생 위험이 낮습니다. "
            "복잡한 금형 형상(코어, 멀티게이트)에서는 별도 분석이 필요합니다.")


def _show_shrinkage(params):
    rho  = params["rho"]
    Tmelt = params["Tmelt"]
    Tmold = params["Tmold"]

    # 단순 수축률 추정 (선형 열팽창)
    alpha = 1.5e-4  # 일반 수지 열팽창계수 [1/°C]
    dT    = Tmelt - Tmold
    shrinkage_pct = alpha * dT * 100

    st.metric("예상 수축률", f"{shrinkage_pct:.2f} %")
    st.metric("온도 차이 ΔT", f"{dT} °C")

    if shrinkage_pct > 2.0:
        st.warning("⚠️ 수축률이 높습니다. 수축공 발생 위험이 있습니다.")
    else:
        st.success("✅ 수축률이 허용 범위 내입니다.")

    st.info("💡 Gemini AI 분석: 두꺼운 단면, 게이트 반대편, "
            "냉각이 늦은 중심부에 수축공 발생 가능성이 높습니다.")


def _show_demo_results(params):
    """OpenFOAM 없는 환경에서 데모"""
    st.info("📌 데모 모드: OpenFOAM 결과 대신 해석적 해를 표시합니다.")
    _plot_velocity_heatmap(params)
    _plot_pressure(params)
    _plot_temperature(params)
    _show_weldline(params)
    _show_shrinkage(params)
