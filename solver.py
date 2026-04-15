import argparse
import os
import json
import numpy as np
import trimesh
import heapq
import time

# ── matplotlib (헤드리스 환경 대응) ──────────────────────────
import matplotlib
matplotlib.use("Agg")  # GUI 없는 서버 환경 필수
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    """
    matplotlib Agg 백엔드로 PNG 저장.
    kaleido / plotly 불필요 — GitHub Actions 헤드리스 환경에서 안정적으로 동작.
    """
    mask = norm_weights <= threshold_ratio
    filled_coords = coords[mask]

    if len(filled_coords) == 0:
        filled_coords = coords[:1]
        color_vals = np.array([0.0])
    else:
        color_vals = norm_weights[mask]
        max_c = max(threshold_ratio, 1e-6)
        color_vals = np.clip(color_vals / max_c, 0.0, 1.0)

    fig = plt.figure(figsize=(10, 7), facecolor="#111111")
    ax = fig.add_subplot(111, projection="3d", facecolor="#111111")

    # 색상: Blues 역방향 (gate=짙은 파랑, front=연한 파랑)
    scatter = ax.scatter(
        filled_coords[:, 0],
        filled_coords[:, 1],
        filled_coords[:, 2],
        c=1.0 - color_vals,       # 역방향
        cmap="Blues",
        s=4,
        alpha=0.85,
        depthshade=False,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label("Flow Distance (gate→front)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("X (mm)", color="white", fontsize=9)
    ax.set_ylabel("Y (mm)", color="white", fontsize=9)
    ax.set_zlabel("Z (mm)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333333")

    ax.set_title(
        f"MIM Fill: {fill_pct:.1f}%  |  Physical Time: {phys_time_label}  |  Frame {frame_idx + 1}",
        color="white", fontsize=12, pad=12,
    )

    # 축 비율 동일하게 (equal aspect)
    all_ranges = coords.max(axis=0) - coords.min(axis=0)
    all_mins   = coords.min(axis=0)
    max_range  = all_ranges.max() / 2.0
    mid        = all_mins + all_ranges / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    img_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.png")
    plt.savefig(img_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
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

    # 2. Physical fill time — reference label ONLY
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

    print(f"[Solver] Done in {elapsed:.1f}s. {num_frames} frames saved to {frames_dir}/")


if __name__ == "__main__":
    main()
