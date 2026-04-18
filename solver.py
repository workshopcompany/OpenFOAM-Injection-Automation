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
    p.add_argument("--gate_pos",    type=str,   default="")   # "x,y,z,dia"
    p.add_argument("--gate_x",      type=float, default=0.0)
    p.add_argument("--gate_y",      type=float, default=0.0)
    p.add_argument("--gate_z",      type=float, default=0.0)
    p.add_argument("--gate_dia",    type=float, default=2.0)
    p.add_argument("--vel_mms",     type=float, default=25.0)
    p.add_argument("--etime",       type=float, default=10.0)
    p.add_argument("--num_frames",  type=int,   default=20)
    
    # [핵심 수정 포인트] float 대신 str로 받아서 "0.5,28.0" 에러 방지
    p.add_argument("--mesh_res_mm", type=str,   default="0.5") 
    
    p.add_argument("--stl_path",    type=str,   default="part.stl")
    p.add_argument("--sim_opts",    type=str,   default="")   # "material,frames,res,screw_dia"
    p.add_argument("--material",    type=str,   default="17-4PH")
    p.add_argument("--screw_dia",   type=float, default=28.0) # 스크류 직경 (mm)
    p.add_argument("--viscosity",   type=float, default=4e-3)
    p.add_argument("--density",     type=float, default=7780)
    p.add_argument("--melt_temp",   type=float, default=185)
    p.add_argument("--temp",        type=float, default=185)
    p.add_argument("--press",       type=float, default=110)
    args = p.parse_args()

    # [핵심 수정 포인트] YAML 파일 한계로 인해 "0.5,28.0" 으로 묶여서 들어오는 문자열을 분리
    if isinstance(args.mesh_res_mm, str):
        if "," in args.mesh_res_mm:
            parts = args.mesh_res_mm.split(",")
            args.mesh_res_mm = float(parts[0].strip())   # 0.5
            args.screw_dia   = float(parts[1].strip())   # 28.0
        else:
            args.mesh_res_mm = float(args.mesh_res_mm)

    # gate_pos 파싱: "x,y,z,dia"
    if args.gate_pos.strip():
        try:
            parts = [v.strip() for v in args.gate_pos.split(",")]
            if len(parts) >= 3:
                args.gate_x, args.gate_y, args.gate_z = float(parts[0]), float(parts[1]), float(parts[2])
            if len(parts) >= 4:
                args.gate_dia = float(parts[3])
            print(f"[Solver] gate_pos -> x={args.gate_x}, y={args.gate_y}, z={args.gate_z}, dia={args.gate_dia}")
        except Exception as e:
            print(f"[Solver] gate_pos parse error: {e}")

    # sim_opts 파싱: "material,frames,res,screw_dia"
    if args.sim_opts.strip():
        try:
            parts = [v.strip() for v in args.sim_opts.split(",")]
            if len(parts) >= 1 and parts[0]: args.material    = parts[0]
            if len(parts) >= 2 and parts[1]: args.num_frames  = int(parts[1])
            # mesh_res_mm 와 screw_dia 는 위에서 안전하게 처리했으므로 통과
            print(f"[Solver] sim_opts -> material={args.material}, frames={args.num_frames}, res={args.mesh_res_mm}, screw={args.screw_dia}mm")
        except Exception as e:
            print(f"[Solver] sim_opts parse error: {e}")

    return args


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


def _export_vtk(all_coords, norm_weights, res):
    """
    복셀 좌표 + Dijkstra 가중치를 VTK UnstructuredGrid로 저장.
    각 복셀을 res×res×res 헥사헤드론으로 출력 → ParaView에서 바로 열림.
    """
    try:
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError:
        print("[Solver] vtk 패키지 없음 — VTK 출력 건너뜀 (pip install vtk)")
        return

    os.makedirs("VTK", exist_ok=True)
    h = res / 2.0  # 복셀 반경

    points_vtk = vtk.vtkPoints()
    grid = vtk.vtkUnstructuredGrid()

    n = len(all_coords)
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(n * 8)

    cell_arr = vtk.vtkCellArray()
    offsets = [0]

    for i, (cx, cy, cz) in enumerate(all_coords):
        base = i * 8
        # 헥사헤드론 8 꼭짓점 (VTK HEX 순서)
        corners = [
            (cx-h, cy-h, cz-h), (cx+h, cy-h, cz-h),
            (cx+h, cy+h, cz-h), (cx-h, cy+h, cz-h),
            (cx-h, cy-h, cz+h), (cx+h, cy-h, cz+h),
            (cx+h, cy+h, cz+h), (cx-h, cy+h, cz+h),
        ]
        for j, (x, y, z) in enumerate(corners):
            pts.SetPoint(base + j, x, y, z)

        hex_cell = vtk.vtkHexahedron()
        for j in range(8):
            hex_cell.GetPointIds().SetId(j, base + j)
        grid.InsertNextCell(hex_cell.GetCellType(), hex_cell.GetPointIds())

    grid.SetPoints(pts)

    # flow_distance 스칼라 (alpha 대응 — 0=gate, 1=최원단)
    flow_arr = numpy_to_vtk(norm_weights, deep=True)
    flow_arr.SetName("flow_distance")
    grid.GetCellData().AddArray(flow_arr)
    grid.GetCellData().SetActiveScalars("flow_distance")

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName("VTK/internal.vtu")
    writer.SetInputData(grid)
    writer.Write()
    print(f"[Solver] ✅ VTK 저장 완료: VTK/internal.vtu ({n} cells)")


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
    # fill() — 내부 볼륨까지 채움 (없으면 표면 shell만 복셀화됨)
    voxel_grid = voxel_grid.fill()
    all_coords = voxel_grid.points
    total_voxels = len(all_coords)
    print(f"[Solver] Voxels: {total_voxels} at res={res}mm (solid fill)")

    # 2. Physical fill time (스크류 면적 기준 유량 보정)
    vol_mm3      = total_voxels * (res ** 3)
    screw_area   = np.pi * (args.screw_dia / 2) ** 2    # mm² — 스크류 단면적
    flow_rate    = screw_area * args.vel_mms if args.vel_mms > 0 else 1.0  # mm³/s
    theo_fill_time = vol_mm3 / flow_rate
    print(f"[Solver] Volume: {vol_mm3:.1f} mm³ | Screw ø{args.screw_dia}mm | Flow: {flow_rate:.0f} mm³/s | Theo fill: {theo_fill_time:.3f}s")

    # 3. Geometric Dijkstra — purely visual ordering
    gate_pos = np.array([args.gate_x, args.gate_y, args.gate_z])

    # 게이트가 (0,0,0) 기본값 그대로이거나 voxel 범위 밖이면
    # → 파트 bounding box 최솟값 면의 centroid로 자동 보정
    bb_min = all_coords.min(axis=0)
    bb_max = all_coords.max(axis=0)
    gate_in_range = np.all(gate_pos >= bb_min - res) and np.all(gate_pos <= bb_max + res)
    # np.allclose 제거: 원점 근처의 유효 게이트(Bottom-Center 등)를 잘못 보정하던 버그 수정
    if not gate_in_range:
        z_min_mask = all_coords[:, 2] < bb_min[2] + res * 2
        gate_pos = all_coords[z_min_mask].mean(axis=0)
        print(f"[Solver] Gate out of range -> fallback to bottom-center: {gate_pos.round(2)}")
    else:
        print(f"[Solver] Gate accepted: {gate_pos.round(3)}")

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

    # --- 여기서부터 추가/수정 시작 ---
    phys_times = norm_weights * theo_fill_time
    pressures = args.press * (1.0 - norm_weights)
    vtu_filename = f"simulation-{args.signal_id}.vtu"
    
    # 위에서 만든 함수 실행
    save_full_solid_mesh_vtu(vtu_filename, all_coords, phys_times, pressures)
    # --- 여기까지 추가/수정 끝 ---

    results = {
        "Signal ID":            args.signal_id,
        "Material":             args.material,
        "Total Voxels":         total_voxels,
        "Part Volume (mm3)":    round(vol_mm3, 2),
        "Gate Dia (mm)":        args.gate_dia,
        "Gate Pos (mm)":        [round(float(v), 3) for v in gate_pos],
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

    # ── VTK 출력 (ParaView / Streamlit pyvista 호환) ──────────────
    _export_vtk(all_coords, norm_weights, res)

    print(f"[Solver] Done in {elapsed:.1f}s. {num_frames} frames saved to {frames_dir}/")


if __name__ == "__main__":
    main()
def save_full_solid_mesh_vtu(filename, centers, fill_times, pressures):
    import base64, zlib, struct
    num_points = len(centers)
    
    def encode_data(data):
        compressed = zlib.compress(data.astype(np.float32).tobytes())
        header = struct.pack("<IIII", 1, len(data)*4, len(data)*4, len(compressed))
        return base64.b64encode(header + compressed).decode('ascii')

    enc_time = encode_data(fill_times)
    enc_press = encode_data(pressures)
    enc_pos = encode_data(centers.flatten())

    vtu_content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{num_points}" NumberOfCells="{num_points}">
      <PointData Scalars="fill_time">
        <DataArray type="Float32" Name="fill_time" format="appended" offset="0" />
        <DataArray type="Float32" Name="pressure" format="appended" offset="{len(enc_time)+4}" />
      </PointData>
      <Points><DataArray type="Float32" Name="Points" NumberOfComponents="3" format="appended" offset="{len(enc_time)+len(enc_press)+8}" /></Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">{" ".join(map(str, range(num_points)))}</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">{" ".join(map(str, range(1, num_points + 1)))}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{" ".join(["1"] * num_points)}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
  <AppendedData encoding="base64">_{enc_time}{enc_press}{enc_pos}</AppendedData>
</VTKFile>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(vtu_content)
