import os
import sys

try:
    from paraview.simple import *
    print("CHECK: ParaView modules loaded successfully.")
except ImportError as e:
    print(f"ERROR: Module import failed: {e}")
    sys.exit(1)

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'results')
case_file = os.path.join(base_dir, 'case.foam')

print(f"CHECK: Base Dir: {base_dir}")
print(f"CHECK: Output Dir: {output_dir}")

if not os.path.exists(case_file):
    print(f"ERROR: {case_file} is missing!")
    sys.exit(1)

# PFOAMReader → OpenFOAMReader 로 변경
try:
    case_foam = OpenFOAMReader(FileName=case_file)
    print("CHECK: OpenFOAMReader loaded.")
except Exception as e:
    print(f"ERROR: OpenFOAMReader failed: {e}")
    sys.exit(1)

case_foam.MeshRegions = ['internalMesh']
case_foam.CellArrays = ['U', 'p_rgh', 'alpha.water']
case_foam.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]
view.Background = [0.1, 0.1, 0.1]

timesteps = case_foam.TimestepValues
if not timesteps:
    print("ERROR: No timesteps found!")
    sys.exit(1)
print(f"CHECK: Found {len(timesteps)} timesteps: {timesteps}")

display = Show(case_foam, view)

# alpha.water (계면) 컬러맵
try:
    ColorBy(display, ('CELLS', 'alpha.water'))
    lut = GetColorTransferFunction('alpha.water')
    lut.RescaleTransferFunction(0.0, 1.0)
    print("CHECK: ColorBy alpha.water success.")
except Exception as e:
    print(f"WARN: alpha.water coloring failed, falling back to U: {e}")
    ColorBy(display, ('POINTS', 'U'))

ResetCamera()

# 마지막 5개 타임스텝 스크린샷
sample_steps = timesteps[-5:] if len(timesteps) >= 5 else timesteps
for i, t in enumerate(sample_steps):
    view.ViewTime = t
    case_foam.UpdatePipeline(t)
    Render()
    img_path = os.path.join(output_dir, f"step_{i+1:02d}_t{t:.3f}.png")
    SaveScreenshot(img_path, view)
    if os.path.exists(img_path):
        print(f"CHECK: Saved {os.path.basename(img_path)}")
    else:
        print(f"ERROR: Failed to save {img_path}")

# VTP 3D 모델 export
try:
    surf = ExtractSurface(Input=case_foam)
    vtp_path = os.path.join(output_dir, "Interactive_Model.vtp")
    SaveData(vtp_path, proxy=surf)
    if os.path.exists(vtp_path):
        print("CHECK: 3D Model (VTP) exported.")
    else:
        print("ERROR: VTP export failed.")
except Exception as e:
    print(f"ERROR: VTP export exception: {e}")

print("DONE: All post-processing finished.")
