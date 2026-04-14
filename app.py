"""
MIM-Ops Pro v2.6 (Integrated)
=============================
핵심 변경 사항:
  1. Geometry & Gate UI: app_animation_claude2.py 스타일로 복구 (Snap 기능 포함)
  2. 해상도 강화: 최솟 두께 / 10 적용 (사용자 요청 반영)
  3. 물리 엔진 유지: Dijkstra 기반 Fountain Flow 로직 (v2.5) 유지
  4. 사이드바 연동: 사이드바의 게이트 및 설정값이 결과물 시뮬레이션에 실시간 반영
"""

import streamlit as st
import os, time, uuid, requests, shutil
from datetime import datetime
import numpy as np
import zipfile
import io
import glob
import re
import pyvista as pv
import plotly.graph_objects as go
import trimesh
from collections import deque
import heapq
from scipy.ndimage import distance_transform_edt

# ═══════════════════════════════════════════════════════════
#  설정 및 초기화
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

# 세션 상태 초기화 (사이드바 연동 변수들)
_init("gx", 0.0); _init("gy", 0.0); _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp", 230.0); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 0.5)
_init("sim_running", False); _init("sim_status", "idle")
_init("sim_logs", [])
_init("last_signal_id", None)
_init("mesh", None)
_init("props", None); _init("props_confirmed", False)
_init("process_confirmed", False)
_init("mat_name", "PA66+30GF")
_init("last_vel_mms", 80.0); _init("last_etime", 0.5)
_init("animation_playing", False); _init("current_frame", 0)
_init("vtk_files", [])
_init("voxel_cache", None)

def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["sim_logs"].append(f"[{ts}] {msg}")

# ═══════════════════════════════════════════════════════════
#  물리 기반 유동 엔진 (Dijkstra) - 핵심 로직 유지
# ═══════════════════════════════════════════════════════════

def compute_voxel_res_mm(mold_trimesh):
    """최솟 두께 / 10 (간소화 방지, 정밀도 향상)"""
    bb = mold_trimesh.bounds
    dims = np.array(bb[1]) - np.array(bb[0])
    valid = dims[dims > 0.1]
    min_dim = float(np.min(valid)) if len(valid) else 10.0
    # 기존 /5에서 /10으로 정밀도 상향
    return float(np.clip(min_dim / 10.0, 0.1, 2.0))

def physics_based_fill_order(occupied, start_vox):
    """v2.5의 다익스트라 기반 Fountain Flow 모사 엔진"""
    Nx, Ny, Nz = occupied.shape
    sx, sy, sz = map(int, start_vox)

    if not occupied[sx, sy, sz]:
        occ_idx = np.argwhere(occupied)
        if len(occ_idx) == 0: return []
        dists = np.sum((occ_idx - np.array([sx, sy, sz]))**2, axis=1)
        sx, sy, sz = map(int, occ_idx[np.argmin(dists)])

    dist_to_wall = distance_transform_edt(occupied)
    min_costs = np.full(occupied.shape, np.inf)
    min_costs[sx, sy, sz] = 0.0
    pq = [(0.0, (sx, sy, sz))]
    order = []
    visited = np.zeros_like(occupied, dtype=bool)

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0: continue
                offsets.append((dx, dy, dz, np.sqrt(dx**2 + dy**2 + dz**2)))

    while pq:
        cost, (cx, cy, cz) = heapq.heappop(pq)
        if visited[cx, cy, cz]: continue
        visited[cx, cy, cz] = True
        order.append((cx, cy, cz))
        for dx, dy, dz, step_dist in offsets:
            nx, ny, nz = cx+dx, cy+dy, cz+dz
            if 0 <= nx < Nx and 0 <= ny < Ny and 0 <= nz < Nz:
                if occupied[nx, ny, nz] and not visited[nx, ny, nz]:
                    speed = (dist_to_wall[nx, ny, nz] + 0.5) ** 2.0
                    edge_cost = step_dist / speed
                    new_cost = cost + edge_cost
                    if new_cost < min_costs[nx, ny, nz]:
                        min_costs[nx, ny, nz] = new_cost
                        heapq.heappush(pq, (new_cost, (nx, ny, nz)))
    return order

# ═══════════════════════════════════════════════════════════
#  사이드바 UI: app_animation_claude2.py 스타일로 통합
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("1. Geometry & Gate")
    stl_file = st.file_uploader("Upload Mold STL", type=["stl"])
    
    if stl_file:
        with open("temp_mold.stl", "wb") as f:
            f.write(stl_file.getbuffer())
        if st.session_state["mesh"] is None:
            st.session_state["mesh"] = trimesh.load_mesh("temp_mold.stl")
            add_log("STL Mesh loaded.")
            # 초기 게이트 위치를 모델 중심으로 설정
            ctr = st.session_state["mesh"].centroid
            st.session_state["gx"], st.session_state["gy"], st.session_state["gz"] = ctr

    if st.session_state["mesh"]:
        mesh = st.session_state["mesh"]
        bounds = mesh.bounds
        
        st.subheader("Gate Coordinates (mm)")
        st.session_state["gx"] = st.slider("X", float(bounds[0][0]), float(bounds[1][0]), float(st.session_state["gx"]))
        st.session_state["gy"] = st.slider("Y", float(bounds[0][1]), float(bounds[1][1]), float(st.session_state["gy"]))
        st.session_state["gz"] = st.slider("Z", float(bounds[0][2]), float(bounds[1][2]), float(st.session_state["gz"]))
        
        if st.button("Snap to Surface"):
            p = np.array([[st.session_state["gx"], st.session_state["gy"], st.session_state["gz"]]])
            cp, _, _ = mesh.nearest.on_surface(p)
            st.session_state["gx"], st.session_state["gy"], st.session_state["gz"] = cp[0]
            st.rerun()
            
        st.session_state["gsize"] = st.number_input("Gate Marker Size", 0.5, 10.0, 2.0)
        
    st.divider()
    st.header("2. Process Parameters")
    st.session_state["mat_name"] = st.selectbox("Material", ["PA66+30GF", "Ti-6Al-4V", "Alumina"])
    st.session_state["temp"] = st.number_input("Temperature (°C)", 100.0, 400.0, st.session_state["temp"])
    st.session_state["vel"] = st.number_input("Injection Velocity (mm/s)", 10.0, 500.0, st.session_state["vel"])

# ═══════════════════════════════════════════════════════════
#  메인 화면 및 시뮬레이션 엔진 연동
# ═══════════════════════════════════════════════════════════
col_main, col_log = st.columns([3, 1])

with col_main:
    if st.session_state["mesh"]:
        # Voxel Grid 생성 및 캐싱 (사이드바 변수 gx, gy, gz 반영)
        mold_mesh = st.session_state["mesh"]
        res_mm = compute_voxel_res_mm(mold_mesh)
        gate_pos = [st.session_state["gx"], st.session_state["gy"], st.session_state["gz"]]
        
        # 캐시 확인 및 갱신 로직
        cache = st.session_state["voxel_cache"]
        if (cache is None or abs(cache["res_mm"] - res_mm) > 0.01 or cache["gate"] != tuple(gate_pos)):
            add_log(f"Rebuilding Voxel Grid (Res: {res_mm:.3f}mm)...")
            vox = mold_mesh.voxelized(pitch=res_mm).fill()
            occupied = vox.matrix.copy()
            origin = np.array(vox.translation)
            
            # 게이트를 voxel 인덱스로 변환
            idx = np.round((np.array(gate_pos) - origin) / res_mm).astype(int)
            idx = np.clip(idx, 0, np.array(occupied.shape) - 1)
            
            fill_order = physics_based_fill_order(occupied, idx)
            st.session_state["voxel_cache"] = {
                "occupied": occupied, "origin": origin, "res_mm": res_mm,
                "gate": tuple(gate_pos), "fill_order": fill_order, "total": len(fill_order)
            }
            cache = st.session_state["voxel_cache"]

        # 시뮬레이션 재생 컨트롤
        total_steps = 20
        st.session_state["current_frame"] = st.slider("Flow Progress", 0, total_steps, st.session_state["current_frame"])
        
        # Plotly 시각화
        fig = go.Figure()
        
        # 1. 몰드 외형 (투명)
        v, f = mold_mesh.vertices, mold_mesh.faces
        fig.add_trace(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
                                opacity=0.1, color='gray', name="Mold"))
        
        # 2. 게이트 위치 표시 (사이드바 연동)
        fig.add_trace(go.Scatter3d(x=[st.session_state["gx"]], y=[st.session_state["gy"]], z=[st.session_state["gz"]],
                                   mode='markers', marker=dict(size=st.session_state["gsize"]*2, color='red'), name="Gate"))

        # 3. 내부 충진 (Dijkstra Fill Order 기반)
        if st.session_state["current_frame"] > 0:
            ratio = st.session_state["current_frame"] / total_steps
            n_fill = int(cache["total"] * ratio)
            if n_fill > 0:
                filled_vox = np.array(cache["fill_order"][:n_fill])
                centers = cache["origin"] + (filled_vox + 0.5) * cache["res_mm"]
                # 간소화 없이 3D Mesh3d로 렌더링 (Solid 효과)
                fig.add_trace(go.Scatter3d(x=centers[:,0], y=centers[:,1], z=centers[:,2],
                                           mode='markers', marker=dict(size=3, color='blue', symbol='square'),
                                           opacity=0.6, name="Fluid"))

        fig.update_layout(scene=dict(aspectmode='data'), height=700, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

with col_log:
    st.subheader("Simulation Logs")
    for log in reversed(st.session_state["sim_logs"]):
        st.text(log)
