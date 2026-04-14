import streamlit as st
import os, time, uuid, shutil, glob, re, heapq
from datetime import datetime
import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt

# ───────────────────── 기본 설정 ─────────────────────
st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

# Session state 초기화 함수
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0); _init("gy", 0.0); _init("gz", 0.0)
_init("gsize", 2.0); _init("temp", 230.0); _init("vel", 80.0); _init("etime", 1.0)
_init("sim_logs", []); _init("mesh", None); _init("mat_name", "PA66+30GF")
_init("num_frames", 10)  # [요청] 기본값 10
_init("current_frame", 0); _init("voxel_cache", None)

def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["sim_logs"].append(f"[{ts}] {msg}")

# ───────────────────── Voxel & Physics Engine ─────────────────────
def compute_voxel_res_mm(mold_trimesh):
    """[요청] 해상도를 1/10로 강화 (더 부드러운 유동)"""
    bb = mold_trimesh.bounds
    dims = np.array(bb[1]) - np.array(bb[0])
    valid = dims[dims > 0.1]
    min_dim = float(np.min(valid)) if len(valid) else 10.0
    return float(np.clip(min_dim / 10.0, 0.1, 2.0))

def build_voxel_grid(mold_trimesh, res_mm):
    vox = mold_trimesh.voxelized(pitch=res_mm).fill()
    return vox.matrix.copy(), np.array(vox.translation, dtype=float)

def physics_based_fill_order(occupied, start_vox):
    Nx, Ny, Nz = occupied.shape
    sx, sy, sz = map(int, start_vox)
    # 시작점이 비어있으면 가장 가까운 채워진 셀 찾기
    if not occupied[sx, sy, sz]:
        occ_idx = np.argwhere(occupied)
        if len(occ_idx) == 0: return []
        sx, sy, sz = occ_idx[np.argmin(np.sum((occ_idx - [sx,sy,sz])**2, axis=1))]

    dist_to_wall = distance_transform_edt(occupied)
    min_costs = np.full(occupied.shape, np.inf)
    min_costs[sx, sy, sz] = 0.0
    pq, order, visited = [(0.0, (sx, sy, sz))], [], np.zeros_like(occupied, dtype=bool)
    offsets = [(dx, dy, dz, np.sqrt(dx**2+dy**2+dz**2)) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==dy==dz==0)]

    while pq:
        cost, (cx, cy, cz) = heapq.heappop(pq)
        if visited[cx, cy, cz]: continue
        visited[cx, cy, cz] = True
        order.append((cx, cy, cz))
        for dx, dy, dz, d_dist in offsets:
            nx, ny, nz = cx+dx, cy+dy, cz+dz
            if 0<=nx<Nx and 0<=ny<Ny and 0<=nz<Nz and occupied[nx,ny,nz] and not visited[nx,ny,nz]:
                # 벽에서 멀수록(중심일수록) 저항이 낮음
                new_cost = cost + (d_dist / ((dist_to_wall[nx,ny,nz]+0.5)**2))
                if new_cost < min_costs[nx,ny,nz]:
                    min_costs[nx,ny,nz] = new_cost
                    heapq.heappush(pq, (new_cost, (nx,ny,nz)))
    return order

def voxels_to_mesh3d(vox_indices, origin, res_mm, max_vox=8000):
    vox_arr = np.array(vox_indices, dtype=float)
    if len(vox_arr) == 0: return None, None, None, None, None
    if len(vox_arr) > max_vox:
        idx = np.round(np.linspace(0, len(vox_arr)-1, max_vox)).astype(int)
        vox_arr = vox_arr[idx]
    
    N = len(vox_arr)
    h = res_mm * 0.5
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * h
    tri = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]])
    centers = origin + (vox_arr + 0.5) * res_mm
    
    all_pts, all_i, all_j, all_k, all_int = np.empty((N*8,3)), np.empty(N*12), np.empty(N*12), np.empty(N*12), np.empty(N*8)
    for ci in range(N):
        all_pts[ci*8:ci*8+8] = corners + centers[ci]
        all_i[ci*12:ci*12+12], all_j[ci*12:ci*12+12], all_k[ci*12:ci*12+12] = tri[:,0]+ci*8, tri[:,1]+ci*8, tri[:,2]+ci*8
        all_int[ci*8:ci*8+8] = ci / (N-1)
    return all_pts, all_i, all_j, all_k, all_int

# ───────────────────── Sidebar (Control) ─────────────────────
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL", type=["stl"])
    if uploaded:
        if st.session_state["mesh"] is None or uploaded.name != st.session_state.get("last_uploaded"):
            st.session_state["mesh"] = trimesh.load(uploaded, file_type="stl")
            st.session_state["last_uploaded"] = uploaded.name
            st.session_state["voxel_cache"] = None

    st.divider(); st.header("📍 2. Gate Setting")
    g_size = st.number_input("Gate Dia (mm)", 0.5, 10.0, 2.0, key="gsize")
    # 좌표 입력
    ix = st.number_input("X", value=float(st.session_state["gx"]))
    iy = st.number_input("Y", value=float(st.session_state["gy"]))
    iz = st.number_input("Z", value=float(st.session_state["gz"]))
    
    # [수정] 모델 표면에 Gate Snap (좌표 보정)
    if st.session_state["mesh"]:
        snap, _, _ = trimesh.proximity.closest_point(st.session_state["mesh"], [[ix, iy, iz]])
        gx, gy, gz = map(float, snap[0])
    else:
        gx, gy, gz = ix, iy, iz
    
    # 보정된 좌표 저장
    st.session_state.update({"gx": ix, "gy": iy, "gz": iz, "gx_f": gx, "gy_f": gy, "gz_f": gz})

    st.divider(); st.header("⚙️ 3. Simulation")
    # [요청] 기본값 10, 동기화 설정
    num_frames = st.select_slider("Animation Frames", options=[5, 10, 15, 20, 30], value=10, key="num_frames")
    vel = st.number_input("Velocity (mm/s)", 1.0, 500.0, 80.0)
    etime = st.number_input("Max Time (s)", 0.1, 10.0, 1.0)

    if st.button("🚀 Run Cloud Sim", type="primary", use_container_width=True):
        add_log(f"Sim launched: {num_frames} frames, {compute_voxel_res_mm(st.session_state['mesh']):.2f}mm res")

# ───────────────────── Main View ─────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📐 Geometry & Gate Preview")
    if st.session_state["mesh"]:
        v = st.session_state["mesh"].vertices
        f = st.session_state["mesh"].faces
        
        fig = go.Figure()
        # 모델 (Mesh)
        fig.add_trace(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2], 
                                color="lightgray", opacity=0.4, name="Mold"))
        # 게이트 (Red Marker) - 보정된 좌표 사용
        fig.add_trace(go.Scatter3d(x=[st.session_state["gx_f"]], y=[st.session_state["gy_f"]], z=[st.session_state["gz_f"]],
                                   mode="markers", marker=dict(size=8, color="red", symbol="diamond"), name="Gate"))
        
        # [수정] 좌표축 비율 고정 (Geometry 왜곡 방지)
        fig.update_layout(scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                          margin=dict(l=0, r=0, b=0, t=0), height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("STL 파일을 업로드해주세요.")

with col2:
    st.header("📟 Logs")
    for log in st.session_state["sim_logs"][-10:]:
        st.code(log)

# ───────────────────── Flow Animation ─────────────────────
if st.session_state["mesh"]:
    st.divider()
    st.subheader("🌊 Flow Simulation")
    
    # Voxel 데이터 생성 및 캐싱
    res_mm = compute_voxel_res_mm(st.session_state["mesh"])
    gate_pos = (st.session_state["gx_f"], st.session_state["gy_f"], st.session_state["gz_f"])
    
    # 캐시 확인 (해상도나 게이트 위치 바뀌면 갱신)
    if (st.session_state["voxel_cache"] is None or 
        st.session_state["voxel_cache"]["res"] != res_mm or 
        st.session_state["voxel_cache"]["gate"] != gate_pos):
        
        occ, origin = build_voxel_grid(st.session_state["mesh"], res_mm)
        g_vox = np.round((np.array(gate_pos) - origin) / res_mm).astype(int).clip(0, np.array(occ.shape)-1)
        order = physics_based_fill_order(occ, g_vox)
        st.session_state["voxel_cache"] = {"order": order, "origin": origin, "res": res_mm, "gate": gate_pos, "total": len(order)}

    cache = st.session_state["voxel_cache"]
    # [요청] 사이드바 설정값에 따라 슬라이더 범위 동기화
    total_steps = st.session_state["num_frames"]
    current_step = st.slider("Animation Step", 0, total_steps, value=st.session_state["current_frame"])
    st.session_state["current_frame"] = current_step

    # 현재 단계에 보여줄 Voxel 계산
    n_to_show = int((current_step / total_steps) * cache["total"])
    pts, fi, fj, fk, intensity = voxels_to_mesh3d(cache["order"][:n_to_show], cache["origin"], cache["res"])

    fig_anim = go.Figure()
    # 투명 몰드
    fig_anim.add_trace(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2], color="gray", opacity=0.1))
    # 유동 (Voxel)
    if pts is not None:
        fig_anim.add_trace(go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], i=fi, j=fj, k=fk, 
                                     intensity=intensity, colorscale="Jet", opacity=0.8))
    
    fig_anim.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig_anim, use_container_width=True)
    
    if n_to_show > 0:
        st.success(f"Filling: {(current_step/total_steps)*100:.1f}% | Active Voxels: {n_to_show:,}")
