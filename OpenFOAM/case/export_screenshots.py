try:
    from paraview.simple import *
except ImportError:
    print("Error: paraview.simple module not found. Check your PYTHONPATH.")
    exit(1)

import os

# 결과 저장용 폴더 (Allrun에서 이미 생성함)
output_dir = 'results'

# --- 1. 데이터 로드 및 뷰 초기 설정 ---
case_file = 'case.foam'
if not os.path.exists(case_file):
    print(f"Error: {case_file} not found.")
    exit(1)

case_foam = PFOAMReader(FileName=case_file)

# 뷰 설정
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]
view.OrientationAxesVisibility = 1 

# --- 2. 파이프라인 업데이트 ---
case_foam.UpdatePipeline()

# --- 3. 가시성 및 컬러링 설정 ---
display = Show(case_foam, view)
ColorBy(display, ('POINTS', 'U'))
u_lut = GetColorTransferFunction('U')
u_pwf = GetOpacityTransferFunction('U')
display.SetScalarBarVisibility(view, True)

# --- 4. 타임스텝 처리 로직 ---
timesteps = case_foam.TimestepValues

if not timesteps or len(timesteps) == 0:
    print("Error: No timesteps found. Simulation might have failed.")
    exit(1)

total_steps = len(timesteps)
target_count = 5

if total_steps <= target_count:
    target_indices = list(range(total_steps))
else:
    target_indices = [int(i * (total_steps - 1) / (target_count - 1)) for i in range(target_count)]

# --- 5. 스크린샷 순회 저장 ---
for i, idx in enumerate(target_indices):
    t_value = timesteps[idx]
    view.ViewTime = t_value
    case_foam.UpdatePipeline(t_value)
    
    if i == 0:
        view.ResetCamera()
    
    u_lut.RescaleTransferFunctionToDataRange(True, False)
    u_pwf.RescaleTransferFunctionToDataRange(True, False)
    Render()
    
    # results 폴더 안에 저장
    filename = f"{output_dir}/result_{i+1:02d}_t{t_value:.4f}.png"
    SaveScreenshot(filename, view)
    print(f"Captured: {filename}")

# --- 6. 3D 인터랙티브 대시보드 데이터 추출 ---
print("Exporting 3D Web Model to results/ folder...")
view.ViewTime = timesteps[-1]
case_foam.UpdatePipeline(timesteps[-1])

surf = ExtractSurface(Input=case_foam)
# results 폴더 안에 저장
SaveData(f"{output_dir}/Interactive_Model.vtp", proxy=surf)

print("All tasks completed successfully. Results are in the results/ folder.")
