try:
    from paraview.simple import *
except ImportError:
    print("Error: paraview.simple module not found.")
    exit(1)

import os

# --- 0. 경로 최적화 ---
# 현재 스크립트 위치를 기준으로 모든 경로를 절대 경로화합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'results')
case_file = os.path.join(base_dir, 'case.foam')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. 데이터 로드 및 뷰 설정 ---
if not os.path.exists(case_file):
    print(f"Error: {case_file} not found.")
    exit(1)

case_foam = PFOAMReader(FileName=case_file)
case_foam.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]
view.OrientationAxesVisibility = 1 

# --- 2. 가시성 및 컬러링 설정 (속도 U) ---
display = Show(case_foam, view)
ColorBy(display, ('POINTS', 'U'))
u_lut = GetColorTransferFunction('U')
u_pwf = GetOpacityTransferFunction('U')

# 컬러바 디자인 (가독성 보강)
u_lut_bar = GetScalarBar(u_lut, view)
u_lut_bar.Title = 'Velocity (U)'
u_lut_bar.ComponentTitle = 'm/s'
display.SetScalarBarVisibility(view, True)

# --- 3. 타임스텝 샘플링 ---
timesteps = case_foam.TimestepValues
if not timesteps:
    print("Error: No timesteps found.")
    exit(1)

total_steps = len(timesteps)
target_count = 5
indices = [int(i * (total_steps - 1) / (target_count - 1)) for i in range(target_count)]

# --- 4. 스크린샷 저장 루프 ---
for i, idx in enumerate(indices):
    t_val = timesteps[idx]
    view.ViewTime = t_val
    case_foam.UpdatePipeline(t_val)
    
    if i == 0:
        view.ResetCamera()
    
    # 데이터 범위 재조정 (이미지 깨짐 방지)
    u_lut.RescaleTransferFunctionToDataRange(True, False)
    u_pwf.RescaleTransferFunctionToDataRange(True, False)
    Render()
    
    filename = f"result_{i+1:02d}_t{t_val:.4f}.png"
    filepath = os.path.join(output_dir, filename)
    SaveScreenshot(filepath, view)
    print(f"Captured: {filename}")

# --- 5. 3D 인터랙티브 데이터 추출 ---
print("Exporting 3D Interactive Model (.vtp)...")
view.ViewTime = timesteps[-1]
case_foam.UpdatePipeline(timesteps[-1])
surf = ExtractSurface(Input=case_foam)

vtp_path = os.path.join(output_dir, 'Interactive_Model.vtp')
SaveData(vtp_path, proxy=surf)

print(f"Post-processing completed. Files saved in: {output_dir}")
