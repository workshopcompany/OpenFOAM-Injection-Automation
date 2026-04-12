import os
import sys

# [체크포인트] 라이브러리 로드 확인
try:
    from paraview.simple import *
    print("CHECK: ParaView modules loaded successfully.")
except ImportError as e:
    print(f"ERROR: Module import failed: {e}")
    sys.exit(1)

# [체크포인트] 경로 설정 확인
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'results')
case_file = os.path.join(base_dir, 'case.foam')

print(f"CHECK: Base Dir: {base_dir}")
print(f"CHECK: Output Dir: {output_dir}")

# [체크포인트] 데이터 로드 확인
if not os.path.exists(case_file):
    print(f"ERROR: {case_file} is missing!")
    sys.exit(1)

case_foam = PFOAMReader(FileName=case_file)
case_foam.UpdatePipeline()

# 뷰 설정
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]

# [체크포인트] 타임스텝 데이터 확인
timesteps = case_foam.TimestepValues
if not timesteps:
    print("ERROR: No timesteps found in case.foam!")
    sys.exit(1)
print(f"CHECK: Found {len(timesteps)} timesteps.")

# --- 메인 로직 (스크린샷) ---
display = Show(case_foam, view)
ColorBy(display, ('POINTS', 'U'))
u_lut = GetColorTransferFunction('U')

for i, t in enumerate(timesteps[-5:]): # 마지막 5개 샘플링
    view.ViewTime = t
    case_foam.UpdatePipeline(t)
    u_lut.RescaleTransferFunctionToDataRange(True, False)
    Render()
    
    img_name = f"step_{i+1}.png"
    img_path = os.path.join(output_dir, img_name)
    SaveScreenshot(img_path, view)
    if os.path.exists(img_path):
        print(f"CHECK: Saved {img_name}")
    else:
        print(f"ERROR: Failed to save {img_name}")

# --- 메인 로직 (3D VTP) ---
surf = ExtractSurface(Input=case_foam)
vtp_path = os.path.join(output_dir, "Interactive_Model.vtp")
SaveData(vtp_path, proxy=surf)
if os.path.exists(vtp_path):
    print("CHECK: 3D Model Exported.")
else:
    print("ERROR: 3D Model Export failed.")

print("DONE: All post-processing steps finished.")
