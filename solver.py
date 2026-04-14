import argparse
import os
import json
import numpy as np
import trimesh
import plotly.graph_objects as go
import heapq
import time

def parse_args():
    p = argparse.ArgumentParser(description="MIM-Ops Cloud Solver: Visual Flow Optimization")
    p.add_argument("--signal_id", type=str, default="manual")
    p.add_argument("--gate_x", type=float, default=0.0)
    p.add_argument("--gate_y", type=float, default=0.0)
    p.add_argument("--gate_z", type=float, default=0.0)
    p.add_argument("--gate_dia", type=float, default=2.0)
    p.add_argument("--vel_mms", type=float, default=25.0)
    p.add_argument("--etime", type=float, default=10.0) # 최대 제한 시간
    p.add_argument("--num_frames", type=int, default=20)
    p.add_argument("--mesh_res_mm", type=float, default=0.5)
    p.add_argument("--stl_path", type=str, default="part.stl")
    # 기타 재료 파라미터들 (결과 기록용)
    p.add_argument("--material", type=str, default="17-4PH")
    p.add_argument("--viscosity", type=float, default=4e-3)
    p.add_argument("--density", type=float, default=7780)
    p.add_argument("--melt_temp", type=float, default=185)
    p.add_argument("--temp", type=float, default=185)
    p.add_argument("--press", type=float, default=110)
    return p.parse_args()

def save_visual_frame(coords, mask, frame_idx, current_phys_time, out_dir):
    """현재 충진 단계의 복셀들을 렌더링하여 저장"""
    filled_coords = coords[mask]
    if len(filled_coords) == 0: return

    fig = go.Figure(data=[go.Scatter3d(
        x=filled_coords[:,0], y=filled_coords[:,1], z=filled_coords[:,2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.8)
    )])

    fig.update_layout(
        title=f"Fill Progress: {int((frame_idx/20)*100)}% | Time: {current_phys_time:.3f}s",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    img_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.png")
    fig.write_image(img_path, width=800, height=600)

def main():
    args = parse_args()
    start_wall_time = time.time()
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    # 1. STL 로드 및 복셀화
    mesh = trimesh.load(args.stl_path)
    res = args.mesh_res_mm
    voxel_grid = mesh.voxelized(res)
    all_coords = voxel_grid.points
    total_voxels = len(all_coords)

    # 2. 물리적 충진 시간 계산 (숫자 결과용)
    vol_mm3 = total_voxels * (res**3)
    gate_area = np.pi * (args.gate_dia/2)**2
    flow_rate = gate_area * args.vel_mms # mm3/s
    theo_fill_time = vol_mm3 / flow_rate if flow_rate > 0 else 0.0

    # 3. Dijkstra 기반 거리(가중치) 계산 (유동 흐름용)
    gate_pos = np.array([args.gate_x, args.gate_y, args.gate_z])
    dists = np.linalg.norm(all_coords - gate_pos, axis=1)
    start_idx = np.argmin(dists)

    # 거리 가중치 맵 생성
    weights = np.full(total_voxels, np.inf)
    weights[start_idx] = 0
    pq = [(0, start_idx)]
    
    # 단순화를 위한 인접 노드 탐색 (KDTree 활용 추천하나 여기선 거리 기반)
    # 실제로는 solver (1).py 등에 있던 인접 로직을 활용
    while pq:
        d, idx = heapq.heappop(pq)
        if d > weights[idx]: continue
        
        # 주변 복셀 탐색 (거리 res*1.5 이내)
        diff = np.linalg.norm(all_coords - all_coords[idx], axis=1)
        neighbors = np.where((diff > 0) & (diff < res * 1.8))[0]
        
        for n_idx in neighbors:
            new_d = d + diff[n_idx]
            if new_d < weights[n_idx]:
                weights[n_idx] = new_d
                heapq.heappush(pq, (new_d, n_idx))

    # 4. 애니메이션 생성 (핵심: 형상 등분)
    max_weight = np.max(weights[weights != np.inf])
    num_frames = args.num_frames
    
    for f in range(num_frames):
        # 시각적 비율 (0.0 ~ 1.0)
        ratio = (f + 1) / num_frames
        # 이 프레임에 해당하는 물리 시간 매핑
        current_time = ratio * theo_fill_time
        # 가중치 컷오프
        threshold = ratio * max_weight
        mask = weights <= threshold
        
        save_visual_frame(all_coords, mask, f, current_time, frames_dir)
        print(f"Frame {f+1}/{num_frames} generated. (Sim Time: {current_time:.4f}s)")

    # 5. 결과 파일 저장
    elapsed = time.time() - start_wall_time
    results = {
        "Signal ID": args.signal_id,
        "Total Voxels": total_voxels,
        "Part Volume (mm3)": round(vol_mm3, 2),
        "Theo Fill Time (s)": round(theo_fill_time, 4),
        "Solver Time (s)": round(elapsed, 2),
        "Status": "Success"
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    with open("results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
