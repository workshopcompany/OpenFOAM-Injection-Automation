"""
solver.py  ─  MIM-Ops GitHub Actions Solver
=============================================
Runs ON GitHub Actions (2-core CPU, 7GB RAM).
NOT executed by Streamlit.

Usage (called by run_sim.yml):
    python solver.py \
        --signal_id  abc12345  \
        --gate_x     5.0  \
        --gate_y     3.0  \
        --gate_z     0.0  \
        --gate_dia   2.0  \
        --vel_mms    25.0 \
        --etime      8.5  \
        --num_frames 15   \
        --material   17-4PH \
        --viscosity  4e-3 \
        --density    7780 \
        --melt_temp  185  \
        --temp       185  \
        --press      110  \
        --mesh_res_mm 0.5 \
        --stl_path   part.stl

Outputs written to ./frames/frame_XX.png  and  ./results.txt
These are uploaded as GitHub Actions Artifacts.
"""

import argparse
import os
import sys
import json
import numpy as np
import trimesh
import plotly.graph_objects as go
import plotly.io as pio
import heapq
from scipy.ndimage import distance_transform_edt
from datetime import datetime

# ─────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MIM-Ops Voxel Flow Solver")
    p.add_argument("--signal_id",   default="unknown")
    p.add_argument("--gate_x",      type=float, default=0.0)
    p.add_argument("--gate_y",      type=float, default=0.0)
    p.add_argument("--gate_z",      type=float, default=0.0)
    p.add_argument("--gate_dia",    type=float, default=2.0)
    p.add_argument("--vel_mms",     type=float, default=25.0)
    p.add_argument("--etime",       type=float, default=10.0)
    p.add_argument("--num_frames",  type=int,   default=15)
    p.add_argument("--material",    default="17-4PH")
    p.add_argument("--viscosity",   type=float, default=4e-3)
    p.add_argument("--density",     type=float, default=7780.0)
    p.add_argument("--melt_temp",   type=float, default=185.0)
    p.add_argument("--temp",        type=float, default=185.0)
    p.add_argument("--press",       type=float, default=110.0)
    p.add_argument("--mesh_res_mm", type=float, default=0.5)
    p.add_argument("--stl_path",    default="part.stl")
    return p.parse_args()


# ─────────────────────────────────────────────
# Voxel Engine  (same physics as original app)
# ─────────────────────────────────────────────
def compute_voxel_res_mm(mold_trimesh, override_mm=None):
    if override_mm and override_mm > 0:
        return float(override_mm)
    bb   = mold_trimesh.bounds
    dims = np.array(bb[1]) - np.array(bb[0])
    valid = dims[dims > 0.1]
    min_dim = float(np.min(valid)) if len(valid) else 10.0
    return float(np.clip(min_dim / 20.0, 0.10, 1.0))

def build_voxel_grid(mold_trimesh, res_mm):
    print(f"  Building voxel grid  res={res_mm:.3f} mm …")
    vox      = mold_trimesh.voxelized(pitch=res_mm)
    vox      = vox.fill()
    occupied = vox.matrix.copy()
    origin   = np.array(vox.translation, dtype=float)
    print(f"  Voxel grid shape: {occupied.shape}  total={occupied.sum():,}")
    return occupied, origin

def gate_to_voxel(gate_mm, origin, res_mm, shape):
    idx = np.round((np.array(gate_mm) - origin) / res_mm).astype(int)
    return tuple(np.clip(idx, 0, np.array(shape) - 1))

def physics_based_fill_order(occupied, start_vox):
    Nx, Ny, Nz = occupied.shape
    sx, sy, sz = int(start_vox[0]), int(start_vox[1]), int(start_vox[2])

    if not occupied[sx, sy, sz]:
        occ_idx = np.argwhere(occupied)
        if len(occ_idx) == 0:
            return []
        dists   = np.sum((occ_idx - np.array([sx, sy, sz]))**2, axis=1)
        nearest = occ_idx[np.argmin(dists)]
        sx, sy, sz = int(nearest[0]), int(nearest[1]), int(nearest[2])

    dist_to_wall = distance_transform_edt(occupied)
    min_costs    = np.full(occupied.shape, np.inf)
    min_costs[sx, sy, sz] = 0.0

    pq = [(0.0, (sx, sy, sz))]
    order   = []
    visited = np.zeros_like(occupied, dtype=bool)

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz, np.sqrt(dx*dx + dy*dy + dz*dz)))

    while pq:
        cost, (cx, cy, cz) = heapq.heappop(pq)
        if visited[cx, cy, cz]:
            continue
        visited[cx, cy, cz] = True
        order.append((cx, cy, cz))

        for dx, dy, dz, step_dist in offsets:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if 0 <= nx < Nx and 0 <= ny < Ny and 0 <= nz < Nz:
                if occupied[nx, ny, nz] and not visited[nx, ny, nz]:
                    wall_dist = dist_to_wall[nx, ny, nz]
                    speed     = (wall_dist + 0.5) ** 2.0
                    new_cost  = cost + step_dist / speed
                    if new_cost < min_costs[nx, ny, nz]:
                        min_costs[nx, ny, nz] = new_cost
                        heapq.heappush(pq, (new_cost, (nx, ny, nz)))

    return order


# ─────────────────────────────────────────────
# Mesh3D helpers
# ─────────────────────────────────────────────
def voxels_to_mesh3d(vox_indices, origin, res_mm, fill_ratios=None, max_vox=8000):
    vox_arr = np.array(vox_indices, dtype=float)
    N = len(vox_arr)
    if N == 0:
        return None, None, None, None, None

    if N > max_vox:
        idx     = np.round(np.linspace(0, N - 1, max_vox)).astype(int)
        vox_arr = vox_arr[idx]
        if fill_ratios is not None:
            fill_ratios = np.asarray(fill_ratios)[idx]
        N = max_vox

    h       = res_mm * 0.5
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                         [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float) * h
    tri     = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                         [2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]])
    centers = origin + (vox_arr + 0.5) * res_mm

    all_pts = np.empty((N * 8, 3), dtype=float)
    all_i   = np.empty(N * 12, dtype=np.int32)
    all_j   = np.empty(N * 12, dtype=np.int32)
    all_k   = np.empty(N * 12, dtype=np.int32)
    all_int = np.empty(N * 8,  dtype=float)

    for ci in range(N):
        pts      = corners + centers[ci]
        base_pt  = ci * 8
        base_tri = ci * 12
        all_pts[base_pt:base_pt+8]      = pts
        all_i[base_tri:base_tri+12]     = tri[:,0] + base_pt
        all_j[base_tri:base_tri+12]     = tri[:,1] + base_pt
        all_k[base_tri:base_tri+12]     = tri[:,2] + base_pt
        val = float(fill_ratios[ci]) if fill_ratios is not None else ci / max(N-1, 1)
        all_int[base_pt:base_pt+8] = val

    return all_pts, all_i, all_j, all_k, all_int


# ─────────────────────────────────────────────
# Frame renderer
# ─────────────────────────────────────────────
def render_frame(mesh_obj, fill_order, origin, res_mm, frame_idx, total_frames,
                 gate_mm, frame_path):
    n_show      = max(1, int((frame_idx + 1) / total_frames * len(fill_order)))
    shown       = fill_order[:n_show]
    fill_ratios = np.linspace(0.0, 1.0, n_show)

    pts, fi, fj, fk, intensity = voxels_to_mesh3d(shown, origin, res_mm, fill_ratios)

    v, f = mesh_obj.vertices, mesh_obj.faces
    fig  = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=v[:,0], y=v[:,1], z=v[:,2],
        i=f[:,0], j=f[:,1], k=f[:,2],
        opacity=0.10, color="lightgray", name="Mold"
    ))
    fig.add_trace(go.Scatter3d(
        x=[gate_mm[0]], y=[gate_mm[1]], z=[gate_mm[2]],
        mode="markers",
        marker=dict(size=8, color="red", symbol="x"),
        name="Gate"
    ))
    if pts is not None:
        fig.add_trace(go.Mesh3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            i=fi, j=fj, k=fk,
            intensity=intensity,
            colorscale="Jet", cmin=0.0, cmax=1.0,
            opacity=0.90, name=f"Fill {(frame_idx+1)/total_frames*100:.0f}%",
            showscale=(frame_idx == total_frames - 1),
            colorbar=dict(title="Fill Order", thickness=15, len=0.6)
        ))

    fill_pct = (frame_idx + 1) / total_frames * 100
    fig.update_layout(
        title=dict(text=f"Frame {frame_idx+1}/{total_frames}  |  Fill: {fill_pct:.0f}%",
                   font=dict(size=14)),
        scene=dict(aspectmode="data",
                   xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="white",
    )

    pio.write_image(fig, frame_path, format="png", width=900, height=620, scale=1.5)
    print(f"  Saved: {frame_path}  ({fill_pct:.0f}%)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    t0   = datetime.now()
    print("=" * 60)
    print(f"MIM-Ops Solver  |  Signal: {args.signal_id}")
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load STL ──
    if not os.path.exists(args.stl_path):
        print(f"ERROR: STL file not found: {args.stl_path}")
        sys.exit(1)

    print(f"Loading STL: {args.stl_path}")
    mesh_obj = trimesh.load(args.stl_path, force="mesh")
    print(f"  Faces: {len(mesh_obj.faces):,}  Volume: {mesh_obj.volume:.1f} mm³")

    # ── Voxelise ──
    res_mm    = compute_voxel_res_mm(mesh_obj, args.mesh_res_mm)
    occupied, origin = build_voxel_grid(mesh_obj, res_mm)

    gate_mm   = (args.gate_x, args.gate_y, args.gate_z)
    start_vox = gate_to_voxel(gate_mm, origin, res_mm, occupied.shape)
    print(f"Gate voxel: {start_vox}")

    # ── Physics flow ──
    print("Running physics-based fill order (Dijkstra)…")
    fill_order = physics_based_fill_order(occupied, start_vox)
    total_vox  = len(fill_order)
    print(f"  Filled voxels: {total_vox:,}")

    # ── Render frames ──
    os.makedirs("frames", exist_ok=True)
    num_frames = args.num_frames
    print(f"Rendering {num_frames} PNG frames…")
    for i in range(num_frames):
        frame_path = os.path.join("frames", f"frame_{i:03d}.png")
        render_frame(
            mesh_obj, fill_order, origin, res_mm,
            i, num_frames, gate_mm, frame_path
        )

    # ── Write results.txt ──
    t1       = datetime.now()
    elapsed  = (t1 - t0).total_seconds()
    vol_mm3  = abs(mesh_obj.volume)
    area_mm2 = np.pi * ((args.gate_dia / 2.0) ** 2)
    flow_rate = area_mm2 * args.vel_mms if args.vel_mms > 0 else 1.0
    theo_fill = vol_mm3 / flow_rate if flow_rate > 0 else 0.0

    results = {
        "Signal ID":          args.signal_id,
        "Material":           args.material,
        "Viscosity (m²/s)":   f"{args.viscosity:.2e}",
        "Density (kg/m³)":    f"{args.density:.0f}",
        "Melt Temp (°C)":     f"{args.melt_temp:.1f}",
        "Injection Temp (°C)":f"{args.temp:.1f}",
        "Pressure (MPa)":     f"{args.press:.1f}",
        "Velocity (mm/s)":    f"{args.vel_mms:.1f}",
        "End Time (s)":       f"{args.etime:.2f}",
        "Gate Dia (mm)":      f"{args.gate_dia:.1f}",
        "Gate Position":      f"({args.gate_x:.2f}, {args.gate_y:.2f}, {args.gate_z:.2f})",
        "Voxel Res (mm)":     f"{res_mm:.3f}",
        "Total Voxels":       str(total_vox),
        "Part Volume (mm³)":  f"{vol_mm3:.1f}",
        "Theo Fill Time (s)": f"{theo_fill:.2f}",
        "Frames Generated":   str(num_frames),
        "Solver Time (s)":    f"{elapsed:.1f}",
        "Finish Time":        t1.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    # Also write JSON for programmatic parsing
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Solver complete.")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
