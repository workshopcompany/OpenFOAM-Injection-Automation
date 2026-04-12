try:
    from paraview.simple import *
except ImportError:
    print("Error: paraview.simple module not found. Check your PYTHONPATH.")
    exit(1)

import os

# --- 0. 경로 및 환경 최적화 ---
# 스크립트 위치를 기준으로 절대 경로를 설정하여 어디서 실행하든 'results' 폴더를 정확히 찾습니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'results')
case_file = os.path.join(base_dir, 'case.foam')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. 데이터 로드 및 뷰 설정 ---
if not os.path.exists(case_file):
    print(f"Error: {case_file} not found in {base_dir}")
    exit(1)

case_foam = PFOAMReader(FileName=case_file)
case_foam.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]
view.OrientationAxesVisibility = 1 # 축 표시 활성화

# --- 2. 가시성 및 컬러링 (속도 U 기준) ---
display = Show(case_foam, view)
ColorBy(display, ('POINTS', 'U'))

# 컬러바(LUT) 및 투명도(PWF) 설정
u_lut = GetColorTransferFunction('U')
u_pwf = GetOpacityTransferFunction('U')

# 컬러바 디자인 최적화 (가독성 향상)
u_lut_bar = GetScalarBar(u_lut, view)
u_lut_bar.Title = 'Velocity (U)'
u_lut_bar.ComponentTitle = 'Magnitude'
u_lut_bar.TitleFontSize = 12
u_lut_bar.LabelFontSize = 10
u_lut_bar.ScalarBarLength = 0.8
display.SetScalarBarVisibility(view, True)

# --- 3. 타임스텝 샘플링 로직 ---
timesteps = case_foam.TimestepValues

if not timesteps or len(timesteps) == 0:
    print("Error: No timesteps found. Check your OpenFOAM case results.")
    exit(1)

total_steps = len(timesteps)
target_count = 5 # 추출할 사진 개수

if total_steps <= target_count:
    target_indices = list(range(total_steps))
else:
    # 시작과 끝을 포함하여 일정한 간격으로 5개 추출
    target_indices = [int(i * (total_steps - 1) / (target_count - 1)) for i in range(target_count)]

print(f"Processing {total_steps} timesteps. Target indices: {target_indices}")

# --- 4. 스크린샷 순회 저장 ---
for i, idx in enumerate(target_indices):
    t_value = timesteps[idx]
    
    # 해당 시점으로 뷰 및 파이프라인 이동
    view.ViewTime = t_value
    case_foam.UpdatePipeline(t_value)
    
    # 첫 프레임에서 카메라 앵글 최적화
    if i == 0:
        view.ResetCamera()
    
    # [핵심] 현재 타임스텝의 실제 데이터 범위에 맞춰 컬러바 재조정
    # 이전 로그에서 발생했던 'Could not determine array range' 오류를 원천 차단합니다.
    u_lut.RescaleTransferFunctionToDataRange(True, False)
    u_pwf.RescaleTransferFunctionToDataRange(True, False)
    
    # 명시적 렌더링 (가상 디스플레이 xvfb 환경에서 필수)
    Render()
    
    # 결과 저장
    filename = f"result_{i+1:02d}_t{t_value:.4f}.png"
    filepath = os.path.join(output_dir, filename)
    SaveScreenshot(filepath, view)
    print(f"Captured: {filename} at Time={t_value}")

# --- 5. 3D 인터랙티브 대시보드 데이터 추출 ---
# 마지막 시점을 웹 대시보드(.vtp)용으로 추출합니다.
print("Exporting 3D Interactive Model (.vtp)...")
view.ViewTime = timesteps[-1]
case_foam.UpdatePipeline(timesteps[-1])

# ExtractSurface 필터를 통해 표면 격자만 추출 (용량 최적화)
surf = ExtractSurface(Input=case_foam)
vtp_path = os.path.join(output_dir, 'Interactive_Model.vtp')
SaveData(vtp_path, proxy=surf)

print(f"All tasks completed. Results are located in: {output_dir}")
