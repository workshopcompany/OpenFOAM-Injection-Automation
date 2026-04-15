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
    p.add_argument("--signal_id",   type=str,   default="manual")
    p.add_argument("--gate_x",      type=float, default=0.0)
    p.add_argument("--gate_y",      type=float, default=0.0)
    p.add_argument("--gate_z",      type=float, default=0.0)
    p.add_argument("--gate_dia",    type=float, default=2.0)
    p.add_argument("--vel_mms",     type=float, default=25.0)
    p.add_argument("--etime",       type=float, default=10.0)
    p.add_argument("--num_frames",  type=int,   default=20)
    p.add_argument("--mesh_res_mm", type=float, default=0.5)
    p.add_argument("--stl_path",    type=str,   default="part.stl")
    p.add_argument("--material",    type=str,   default="17-4PH")
    p.add_argument("--viscosity",   type=float, default=4e-3)
    p.add_argument("--density",     type=float, default=7780)
    p.add_argument("--melt_temp",   type=float, default=185)
    p.add_argument("--temp",        type=float, default=185)
    p.add_argument("--press",       type=float, default=110)
    return p.parse_args()


def compute_dijkstra_weights(all_coords, start_idx, res):
    """
    Dijkstra BFS on voxel grid.
    Returns normalized weights [0.0 ~ 1.0] where 0 = gate, 1 = farthest point.
    Purely GEOMETRIC — no physical time involved.
    """
    from scipy.spatial import cKDTree
    total = len(all_coords)
    weights = np.full(total, np.inf)
    weights[start_idx] = 0.0
    pq = [(0.0, start_idx)]
    neighbor_radius = res * 1.85  # covers diagonal voxel neighbors

    tree = cKDTree(all_coords)

    while pq:
        d, idx = heapq.heappop(pq)
        if d > weights[idx]:
            continue
        neighbor_indices = tree.query_ball_point(all_coords[idx], neighbor_radius)
        for n_idx in neighbor_indices:
            if n_idx == idx:
                continue
            dist = float(np.linalg.norm(all_coords[idx] - all_coords[n_idx]))
            new_d = d + dist
            if new_d < weights[n_idx]:
                weights[n_idx] = new_d
                heapq.heappush(pq, (new_d, n_idx))

    finite_mask = weights != np.inf
    max_w = float(np.max(weights[finite_mask])) if finite_mask.any() else 1.0
    weights[~finite_mask] = max_w
    return weights / max_w


def save_visual_frame(coords, norm_weights, threshold_ratio, frame_idx,
                      phys_time_label, fill_pct, out_dir):
    mask = norm_weights <= threshold_ratio
    filled_coords = coords[mask]

    if len(filled_coords) == 0:
        filled_coords = coords[:1]
        color_vals = [0.0]
    else:
        color_vals = norm_weights[mask]
        max_c = max(threshold_ratio, 1e-6)
        color_vals = np.clip(color_vals / max_c, 0, 1).tolist()

    fig = go.Figure(data=[go.Scatter3d(
        x=filled_coords[:, 0],
        y=filled_coords[:, 1],
        z=filled_coords[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=color_vals,
            colorscale='Blues',
            reversescale=True,
            opacity=0.85,
            colorbar=dict(title="Flow Distance", thickness=10, tickfont=dict(color='white'))
        )
    )])

    fig.update_layout(
        title=dict(
            text=(f"MIM Fill: {fill_pct:.1f}%  |  "
                  f"Physical Time: {phys_time_label}  |  "
                  f"Frame {frame_idx + 1}"),
            font=dict(color='white', size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (mm)', color='white'),
            yaxis=dict(title='Y (mm)', color='white'),
            zaxis=dict(title='Z (mm)', color='white'),
            bgcolor='#111111',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='#111111',
    )

    img_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.png")
    fig.write_image(img_path, width=900, height=650)
    return img_path


def main():
    args = parse_args()
    start_wall_time = time.time()
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    print(f"[Solver] STL: {args.stl_path}")
    print(f"[Solver] Gate: ({args.gate_x}, {args.gate_y}, {args.gate_z}), dia={args.gate_dia}mm")

    # 1. Load & voxelise STL
    mesh = trimesh.load(args.stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    res = args.mesh_res_mm
    voxel_grid = mesh.voxelized(res)
    all_coords = voxel_grid.points
    total_voxels = len(all_coords)
    print(f"[Solver] Voxels: {total_voxels} at res={res}mm")

    # 2. Physical fill time — reference label ONLY (not used for animation pacing)
    vol_mm3 = total_voxels * (res ** 3)
    gate_area = np.pi * (args.gate_dia / 2) ** 2
    flow_rate = gate_area * args.vel_mms if args.vel_mms > 0 else 1.0
    theo_fill_time = vol_mm3 / flow_rate  # seconds
    print(f"[Solver] Volume: {vol_mm3:.1f} mm³ | Theo fill time: {theo_fill_time:.4f}s")

    # 3. Geometric Dijkstra — purely visual ordering
    gate_pos = np.array([args.gate_x, args.gate_y, args.gate_z])
    dists_to_gate = np.linalg.norm(all_coords - gate_pos, axis=1)
    start_idx = int(np.argmin(dists_to_gate))
    print(f"[Solver] Nearest gate voxel: idx={start_idx}")
    print("[Solver] Running Dijkstra BFS...")
    norm_weights = compute_dijkstra_weights(all_coords, start_idx, res)
    print("[Solver] Dijkstra complete.")

    # 4. Animation frames
    # DESIGN:
    #   visual_ratio = (f+1)/num_frames  — uniform geometric steps (0.0→1.0)
    #   fill_pct     = visual_ratio * 100
    #   phys_label   = visual_ratio * theo_fill_time  — annotation only
    num_frames = args.num_frames
    print(f"[Solver] Generating {num_frames} frames...")

    for f in range(num_frames):
        visual_ratio = (f + 1) / num_frames
        fill_pct     = visual_ratio * 100.0
        phys_time    = visual_ratio * theo_fill_time
        if phys_time < 1.0:
            phys_label = f"{phys_time * 1000:.1f} ms"
        else:
            phys_label = f"{phys_time:.3f} s"

        save_visual_frame(
            coords=all_coords,
            norm_weights=norm_weights,
            threshold_ratio=visual_ratio,
            frame_idx=f,
            phys_time_label=phys_label,
            fill_pct=fill_pct,
            out_dir=frames_dir,
        )
        print(f"  Frame {f+1}/{num_frames} | Fill: {fill_pct:.1f}% | t={phys_label}")

    # 5. Save results
    elapsed = time.time() - start_wall_time
    results = {
        "Signal ID":            args.signal_id,
        "Material":             args.material,
        "Total Voxels":         total_voxels,
        "Part Volume (mm3)":    round(vol_mm3, 2),
        "Gate Dia (mm)":        args.gate_dia,
        "Injection Vel (mm/s)": args.vel_mms,
        "Theo Fill Time (s)":   round(theo_fill_time, 4),
        "Num Frames":           num_frames,
        "Mesh Res (mm)":        res,
        "Solver Time (s)":      round(elapsed, 2),
        "Status":               "Success",
        "Note": (
            "Frames are geometry-driven (Dijkstra). "
            "Physical time is a proportional label — decoupled from animation speed."
        ),
    }

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=4)

    with open("results.txt", "w") as fh:
        for k, v in results.items():
            fh.write(f"{k}: {v}\n")

    print(f"[Solver] Done in {elapsed:.1f}s. Frames in {frames_dir}/")


if __name__ == "__main__":
    main()
