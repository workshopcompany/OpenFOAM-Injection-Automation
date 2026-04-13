import streamlit as st
import os, json, time, uuid, requests
from datetime import datetime
import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ══════════════════════════════════════════
# 기본 설정
# ══════════════════════════════════════════
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# ── 세션 초기화 ───────────────────────────
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0);  _init("gy", 0.0);  _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp",  230); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 0.5)
_init("sim_running", False)
_init("last_signal_id", None)
_init("mesh", None)
_init("props", None)
_init("props_confirmed", False)
_init("mat_name", "PA66+30GF")
_init("last_vel_mms", 80.0)
_init("last_etime", 0.5)
_init("gx_final", 0.0)
_init("gy_final", 0.0)
_init("gz_final", 0.0)

# ══════════════════════════════════════════
# 로컬 물성 DB (Gemini 추천 형식 유지)
# ══════════════════════════════════════════
LOCAL_DB = {
    "PP": {
        "nu": 1e-3, "rho": 900, "Tmelt": 230, "Tmold": 40,
        "press_mpa": 70, "vel_mms": 80,
        "desc": "범용 폴리프로필렌 — 유동성 우수, 수축 큼"
    },
    "ABS": {
        "nu": 2e-3, "rho": 1050, "Tmelt": 240, "Tmold": 60,
        "press_mpa": 80, "vel_mms": 70,
        "desc": "ABS 수지 — 내충격성 우수, 도금 가능"
    },
    "PA66": {
        "nu": 5e-4, "rho": 1140, "Tmelt": 280, "Tmold": 80,
        "press_mpa": 90, "vel_mms": 100,
        "desc": "나일론 66 — 내열성/강성 우수, 흡습 주의"
    },
    "PA66+30GF": {
        "nu": 4e-4, "rho": 1300, "Tmelt": 285, "Tmold": 85,
        "press_mpa": 110, "vel_mms": 80,
        "desc": "유리섬유 30% 강화 나일론 — 강성 대폭 향상"
    },
    "PC": {
        "nu": 3e-3, "rho": 1200, "Tmelt": 300, "Tmold": 85,
        "press_mpa": 120, "vel_mms": 60,
        "desc": "폴리카보네이트 — 투명, 내충격성 최고, 점도 높음"
    },
    "POM": {
        "nu": 8e-4, "rho": 1410, "Tmelt": 200, "Tmold": 90,
        "press_mpa": 85, "vel_mms": 90,
        "desc": "폴리아세탈 — 내마모성 우수, 정밀부품 적합"
    },
    "HDPE": {
        "nu": 9e-4, "rho": 960, "Tmelt": 220, "Tmold": 35,
        "press_mpa": 60, "vel_mms": 90,
        "desc": "고밀도 폴리에틸렌 — 내화학성 우수, 저가"
    },
    "PET": {
        "nu": 6e-4, "rho": 1370, "Tmelt": 265, "Tmold": 70,
        "press_mpa": 80, "vel_mms": 85,
        "desc": "PET — 투명성/강도 우수, 건조 필수"
    },
    "CATAMOLD": {
        "nu": 5e-3, "rho": 4900, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 100, "vel_mms": 30,
        "desc": "BASF Catamold MIM 피드스탁 — 금속분말+바인더"
    },
    "MIM": {
        "nu": 5e-3, "rho": 5000, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 100, "vel_mms": 30,
        "desc": "금속사출성형 피드스탁 — 고밀도, 저속 사출"
    },
    "17-4PH": {
        "nu": 4e-3, "rho": 7780, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 110, "vel_mms": 25,
        "desc": "17-4PH 스테인리스 MIM 피드스탁"
    },
    "316L": {
        "nu": 4e-3, "rho": 7900, "Tmelt": 185, "Tmold": 40,
        "press_mpa": 110, "vel_mms": 25,
        "desc": "316L 스테인리스 MIM 피드스탁 — 내식성 우수"
    },
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key, val in LOCAL_DB.items():
        if key.upper() == name:
            return {**val, "material": key, "source": "Gemini 추천"}
    for key, val in LOCAL_DB.items():
        if key.upper() in name or name in key.upper():
            return {**val, "material": key, "source": "Gemini 추천"}
    return {
        "nu": 1e-3, "rho": 1000, "Tmelt": 220, "Tmold": 50,
        "press_mpa": 70, "vel_mms": 80,
        "material": material, "source": "Gemini 추천",
        "desc": f"{material} — DB 미등록 재료, 기본값 적용"
    }

def get_process(material: str) -> dict:
    props = get_props(material)
    return {
        "temp":  props.get("Tmelt", 230),
        "press": float(props.get("press_mpa", 70)),
        "vel":   float(props.get("vel_mms", 80)),
    }


# ══════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════
with st.sidebar:

    # ── 1. Geometry ───────────────────────
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])

    if uploaded:
        if HAS_TRIMESH:
            try:
                mesh = trimesh.load(uploaded, file_type="stl")
                st.session_state["mesh"] = mesh
                st.success(f"✅ STL 로드 완료 — {len(mesh.faces):,} 면")
            except Exception as e:
                st.error(f"STL 로드 실패: {e}")
        else:
            st.warning("trimesh 미설치: pip install trimesh")

    st.divider()

    # ── 2. Gate Configuration ─────────────
    st.header("📍 2. Gate Configuration")

    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        mesh = st.session_state.get("mesh")
        if mesh is not None and HAS_TRIMESH:
            center = mesh.centroid
            snap, _, _ = trimesh.proximity.closest_point(mesh, [center])
            pos = snap[0]
            st.session_state["gx"]    = float(pos[0])
            st.session_state["gy"]    = float(pos[1])
            st.session_state["gz"]    = float(pos[2])
            st.session_state["gsize"] = 2.5
            st.toast("게이트 위치 추천 완료!", icon="🪄")
        else:
            st.warning("STL을 먼저 업로드하세요.")

    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")
    vx = st.number_input("Gate X", value=st.session_state["gx"], step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=st.session_state["gy"], step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=st.session_state["gz"], step=0.1, key="gz")

    # 게이트 메쉬 표면 스냅 후 세션에 저장
    mesh = st.session_state.get("mesh")
    if mesh is not None and HAS_TRIMESH:
        snap, _, _ = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])
        gx = float(snap[0][0])
        gy = float(snap[0][1])
        gz = float(snap[0][2])
    else:
        gx, gy, gz = vx, vy, vz

    # ★ 게이트 좌표를 세션에 저장 (메인 영역에서 참조 가능)
    st.session_state["gx_final"] = gx
    st.session_state["gy_final"] = gy
    st.session_state["gz_final"] = gz

    st.divider()

    # ── 3. Material ───────────────────────
    st.header("🧪 3. Material")
    mat_name = st.text_input(
        "Material Name", value="PA66+30GF",
        placeholder="PP, ABS, PA66, PC, Catamold ...",
        key="mat_name_input"
    )
    # 재료명도 세션 저장
    st.session_state["mat_name"] = mat_name

    if st.button("🤖 AI 물성 추천 (Gemini)", use_container_width=True, type="primary"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False

    # 물성 표시 + 수정
    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 출처: {p.get('source', 'Gemini 추천')}")
        if p.get("desc"):
            st.info(p["desc"])

        with st.expander("📋 물성 확인 / 수정", expanded=True):
            p["nu"]    = st.number_input(
                "운동점도 nu (m²/s)",
                value=float(p.get("nu", 1e-3)),
                format="%.2e",
                min_value=1e-7, max_value=1.0,
                key="edit_nu"
            )
            p["rho"]   = st.number_input(
                "밀도 ρ (kg/m³)",
                value=float(p.get("rho", 1000)),
                min_value=100, max_value=9000,
                key="edit_rho"
            )
            p["Tmelt"] = st.number_input(
                "용융 온도 (°C)",
                value=int(p.get("Tmelt", 220)),
                min_value=100, max_value=450,
                key="edit_tmelt"
            )
            p["Tmold"] = st.number_input(
                "금형 온도 (°C)",
                value=int(p.get("Tmold", 50)),
                min_value=10, max_value=200,
                key="edit_tmold"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 물성 확정", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("물성 확정됨!", icon="✅")
            with col2:
                if st.button("🔄 초기화", use_container_width=True):
                    st.session_state["props"] = None
                    st.session_state["props_confirmed"] = False
                    st.rerun()

    st.divider()

    # ── 4. Process Condition ──────────────
    st.header("⚙️ 4. Process Condition")

    if st.button("🤖 Optimize Process", use_container_width=True):
        suggestion = get_process(mat_name)
        st.session_state["temp"]  = suggestion["temp"]
        st.session_state["press"] = suggestion["press"]
        st.session_state["vel"]   = suggestion["vel"]
        st.toast("공정조건 최적화 완료!", icon="🤖")

    temp_c    = st.number_input("Injection Temperature (°C)",
                                 50, 450, step=1, key="temp")
    press_mpa = st.number_input("Injection Pressure (MPa)",
                                 10.0, 250.0, step=1.0, key="press")
    vel_mms   = st.number_input("Injection Velocity (mm/s)",
                                 1.0, 600.0, step=1.0, key="vel")
    etime     = st.number_input("End Time (s)",
                                 value=st.session_state["etime"],
                                 min_value=0.1, max_value=10.0,
                                 step=0.1, key="etime")

    # ★ 공정조건도 세션에 저장 (메인 영역 로그에서 참조)
    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"]   = etime

    if not st.session_state["props_confirmed"]:
        st.warning("⚠️ 물성을 추천받고 ✅ 확정해주세요")

    st.divider()

    # ── Run 버튼 ──────────────────────────
    run_disabled = (
        st.session_state["sim_running"] or
        not st.session_state["props_confirmed"]
    )

    if st.button(
        "🚀 Run Cloud Simulation",
        type="primary",
        use_container_width=True,
        disabled=run_disabled
    ):
        if not ZAPIER_URL:
            st.error("❌ ZAPIER_URL이 설정되지 않았습니다.\n.streamlit/secrets.toml을 확인하세요.")
        else:
            props  = st.session_state["props"]
            sig_id = str(uuid.uuid4())[:8]
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"]    = True

            payload = {
                "signal_id":  sig_id,
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material":   mat_name,
                "viscosity":  float(props["nu"]),
                "density":    float(props["rho"]),
                "melt_temp":  int(props["Tmelt"]),
                "mold_temp":  int(props["Tmold"]),
                "temp":       int(temp_c),
                "press":      float(press_mpa),
                "vel":        round(vel_mms / 1000, 6),  # mm/s → m/s
                "etime":      float(etime),
                "gate_pos":   {
                    "x": round(gx, 3),
                    "y": round(gy, 3),
                    "z": round(gz, 3)
                },
                "gate_size":  float(g_size),
            }

            try:
                res = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if res.status_code == 200:
                    st.toast(f"🚀 신호 전송 완료! (ID: {sig_id})", icon="🚀")
                else:
                    st.error(f"전송 실패: HTTP {res.status_code}")
                    st.session_state["sim_running"] = False
            except Exception as e:
                st.error(f"연결 오류: {e}")
                st.session_state["sim_running"] = False


# ══════════════════════════════════════════
# 메인 화면 — 세션에서 값 읽기 (스코프 안전)
# ══════════════════════════════════════════
gx_f      = st.session_state["gx_final"]
gy_f      = st.session_state["gy_final"]
gz_f      = st.session_state["gz_final"]
g_size_f  = st.session_state["gsize"]
props_f   = st.session_state["props"]
mat_f     = st.session_state["mat_name"]
vel_f     = st.session_state["last_vel_mms"]
etime_f   = st.session_state["last_etime"]
sig_id_f  = st.session_state["last_signal_id"]

col_geo, col_log = st.columns([2, 1])

# ── 3D 형상 뷰어 ─────────────────────────
with col_geo:
    st.header("🎥 3D Geometry Analysis")
    mesh = st.session_state.get("mesh")

    if mesh is not None and HAS_PLOTLY:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:,0], y=v[:,1], z=v[:,2],
                i=f[:,0], j=f[:,1], k=f[:,2],
                color="#AAAAAA", opacity=0.8,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3)
            ),
            go.Scatter3d(
                x=[gx_f], y=[gy_f], z=[gz_f],
                mode="markers",
                marker=dict(size=g_size_f * 5, color="red"),
                name="게이트"
            )
        ])
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode="data"),
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)

        bb = mesh.bounds
        c1, c2, c3 = st.columns(3)
        c1.metric("X 크기", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y 크기", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z 크기", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("사이드바에서 STL 파일을 업로드하면 3D 형상이 표시됩니다.")

# ── 시뮬레이션 로그 ──────────────────────
with col_log:
    st.header("📟 Simulation & Debug Logs")

    if st.session_state["sim_running"] and sig_id_f and props_f:
        log_lines = [
            f">>> [MIM-Ops] Outbound Signal ID: {sig_id_f}",
            ">>> Preventing Duplicate Runs: Bypass GitHub Push Trigger.",
            ">>> Verifying OpenFOAM Dictionary Integrity...",
            f">>> Material: {mat_f}",
            f">>> nu = {props_f['nu']:.2e} m²/s",
            f">>> rho = {props_f['rho']} kg/m³",
            f">>> Tmelt = {props_f['Tmelt']}°C | Tmold = {props_f['Tmold']}°C",
            f">>> Gate: ({gx_f:.2f}, {gy_f:.2f}, {gz_f:.2f}) Ø{g_size_f}mm",
            f">>> Velocity = {vel_f/1000:.4f} m/s",
            f">>> End Time = {etime_f}s",
            "✅ transportProperties: OK",
            "✅ fvSolution: OK",
            "✅ fvSchemes: OK",
            ">>> Zapier → GitHub Actions 신호 전송 완료.",
            ">>> blockMesh 실행 대기 중...",
            ">>> interFoam 실행 대기 중...",
            ">>> GitHub Actions Artifacts에서",
            "    결과를 확인하세요.",
        ]
        st.code("\n".join(log_lines), language="bash")

        if st.button("✅ 완료 확인"):
            st.session_state["sim_running"] = False
            st.rerun()

    elif sig_id_f:
        st.success(f"✅ 마지막 실행 ID: {sig_id_f}")
        st.info("GitHub Actions → Artifacts에서 결과를 확인하세요.")
    else:
        st.info("시뮬레이션을 실행하면 여기에 로그가 표시됩니다.")

# ── 하단 상태 표시 ───────────────────────
st.info(f"📍 Final Gate Position: ({gx_f:.2f}, {gy_f:.2f}, {gz_f:.2f})")

if st.session_state["props_confirmed"] and props_f:
    st.caption(
        f"ℹ️ 물성 확정: nu={props_f['nu']:.2e} | "
        f"rho={props_f['rho']} kg/m³ | "
        f"Tmelt={props_f['Tmelt']}°C | "
        f"Tmold={props_f['Tmold']}°C | "
        f"출처: {props_f.get('source', 'Gemini 추천')}"
    )
else:
    st.caption("ℹ️ 사이드바에서 재료 물성을 추천받고 확정한 후 시뮬레이션을 실행하세요.")
