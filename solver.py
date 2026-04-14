import argparse
import os
import sys
import json
import numpy as np
import trimesh
import plotly.graph_objects as go
import plotly.io as pio
import heapq
import time
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(description="MIM-Ops Voxel Flow Solver (Flow-Ratio Frame Logic)")
    p.add_argument("--signal_id", type=str, default="manual")
    p.add_argument("--gate_x", type=float, default=0.0)
    p.add_argument("--gate_y", type=float, default=0.0)
    p.add_argument("--gate_z", type=float, default=0.0)
    p.add_argument("--gate_dia", type=float, default=2.0)
    p.add_argument("--vel_mms", type=float, default=25.0)
    p.add_argument("--etime", type=float, default=10.0)
    p.add_argument("--num_frames", type=int, default=20)
    p.add_argument("--material", type=str, default="17-4PH")
    p.add_argument("--viscosity", type=float, default=4e-3)
    p.add_argument("--density", type=float, default=7780)
    p.add_argument("--melt_temp", type=float, default=185)
    p.add_argument("--temp", type=float, default=185)
    p.add_argument("--press", type=float, default=110)
    p.add_argument("--mesh_res_mm", type=float, default=0.5)
    p.add_argument("--stl_path", type=str, default="part.stl")
    return p.parse_args()

def save_visual_frame(visited_indices, all_indices, coords, frame_idx, current_time, out_dir):
    """현재까지 채워진 Voxel들을 이미지로 저장"""
    if not visited_indices: return
    
    visited_list = list(visited_indices)
    v_coords = coords[visited_list]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=v_coords[:,0], y=v_coords[:,1], z=v_coords[:,2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.8)
    )])
    
    fig.update_layout(
        title=f"Time: {current_time:.3f}s (Frame {frame_idx})",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    img_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.png")
    fig.write_image(img_path, width=800, height=600)

def main():
    args = parse_args()
    start_wall_time = time.time()
    
    # 디렉토리 준비
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    # 1. STL 로드 및 복셀화
    mesh = trimesh.load(args.stl_path)
    res_mm = args.mesh_res_mm
    voxel_grid = mesh.voxelized(res_mm)
    all_coords = voxel_grid.points
    total_voxels = len(all_coords)
    
    # 2. 게이트에서 가장 가까운 복셀 찾기
    gate_point = np.array([args.gate_x, args.gate_y, args.gate_z])
    dists = np.linalg.norm(all_coords - gate_point, axis=1)
    start_idx = np.argmin(dists)
    
    # 3. 인접 리스트 생성 (Flow Simulation 준비)
    # 간단한 격자 인접 로직 (Dijkstra 기반 유동 해석)
    pq = [(0.0, start_idx)] # (time, index)
    visited = {}
    
    # 프레임 생성을 위한 인터벌 계산 (개수 기반)
    # 시간과 관계없이 전체 복셀을 num_frames 등분함
    frame_interval = max(1, total_voxels // args.num_frames)
    frames_saved = 0
    
    # 실제 물리 유속 계산 (mm/s)
    vel = args.vel_mms if args.vel_mms > 0 else 1.0
    
    print(f"Starting simulation: {total_voxels} voxels, target {args.num_frames} frames.")

    # 4. 메인 솔버 루프
    last_t = 0.0
    while pq and len(visited) < total_voxels:
        t, idx = heapq.heappop(pq)
        
        if idx in visited: continue
        visited[idx] = t
        last_t = t
        
        # [수정 포인트] 충진 개수가 도달할 때마다 프레임 저장
        if len(visited) % frame_interval == 0 and frames_saved < args.num_frames:
            save_visual_frame(visited.keys(), range(total_voxels), all_coords, frames_saved, t, frames_dir)
            frames_saved += 1
            print(f"Frame {frames_saved} saved at t={t:.4f}s ({len(visited)}/{total_voxels} voxels)")

        # 주변 복셀 탐색 (거리/속도로 시간 계산)
        # 이 예시에서는 단순화를 위해 인접한 노드를 t + (거리/속도)로 추가
        # (실제 solver 로직에 맞춰 인접 노드 계산 부분을 유지하세요)
        # ... (생략된 기존 인접 노드 확장 로직) ...
        # 임시 로직:
        for i in range(len(all_coords)):
            if i not in visited:
                d_node = np.linalg.norm(all_coords[idx] - all_coords[i])
                if d_node <= res_mm * 1.5: # 인접 노드 조건
                    heapq.heappush(pq, (t + d_node/vel, i))

    # 최종 결과 기록
    elapsed = time.time() - start_wall_time
    vol_mm3 = total_voxels * (res_mm ** 3)
    
    summary = {
        "Signal ID": args.signal_id,
        "Total Voxels": total_voxels,
        "Part Volume (mm3)": vol_mm3,
        "Last Time Step": round(last_t, 4),
        "Solver Time (s)": round(elapsed, 2),
        "Frames Generated": frames_saved
    }
    
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    with open("results.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
            
    print("Simulation Completed Successfully.")

if __name__ == "__main__":
    main()
