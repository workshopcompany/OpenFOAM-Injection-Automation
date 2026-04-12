try:
    from paraview.simple import *
except ImportError:
    print("Error: paraview.simple module not found. Check your PYTHONPATH.")
    exit(1)

import os

# --- 1. 데이터 로드 및 뷰 초기 설정 ---
case_file = 'case.foam'
if not os.path.exists(case_file):
    print(f"Error: {case_file} not found.")
    exit(1)

case_foam = PFOAMReader(FileName=case_file)

# 뷰 설정 (해상도는 720p 유지)
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]
view.OrientationAxesVisibility = 1 # 축 표시 (전문적인 느낌 추가)

# --- 2. 초기 파이프라인 업데이트 ---
# 이 단계가 명확해야 나중에 'U' 필드를 인식할 때 오류가 없습니다.
case_foam.UpdatePipeline()

# --- 3. 가시성 및 컬러링 설정 (속도 U 기준) ---
display = Show(case_foam, view)
ColorBy(display, ('POINTS', 'U'))

# 컬러바(LUT) 설정 및 초기 업데이트
u_lut = GetColorTransferFunction('U')
u_pwf = GetOpacityTransferFunction('U') # 투명도 함수도 함께 가져옵니다.

# 컬러바 표시 설정
display.SetScalarBarVisibility(view, True)

# --- 4. 타임스텝 처리 로직 ---
timesteps = case_foam.TimestepValues

if not timesteps or len(timesteps) == 0:
    print("Error: No timesteps found. Simulation might have failed.")
    exit(1)

total_steps = len(timesteps)
target_count = 5

# 사진을 찍을 인덱스 계산
if total_steps <= target_count:
    target_indices = list(range(total_steps))
else:
    target_indices = [int(i * (total_steps - 1) / (target_count - 1)) for i in range(target_count)]

print(f"Total timesteps: {total_steps}. Capturing at indices: {target_indices}")

# --- 5. 스크린샷 순회 저장 ---
# 폴더가 없는 경우 대비
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

for i, idx in enumerate(target_indices):
    t_value = timesteps[idx]
    
    # 해당 타임스텝으로 이동 및 강제 업데이트
    view.ViewTime = t_value
    case_foam.UpdatePipeline(t_value)
    
    # 카메라 초기화 (데이터가 차오르는 모습에 따라 앵글을 맞춤)
    if i == 0:
        view.ResetCamera()
    
    # 컬러바 범위를 현재 데이터에 맞춰 재조정 (경고 발생 방지 핵심)
    u_lut.RescaleTransferFunctionToDataRange(True, False)
    u_pwf.RescaleTransferFunctionToDataRange(True, False)
    
    Render()
    
    # 파일명 저장
    filename = f"screenshots/result_{i+1:02d}_t{t_value:.4f}.png"
    SaveScreenshot(filename, view)
    print(f"Captured: {filename} at T={t_value}")

# --- 6. 3D 인터랙티브 대시보드 데이터 추출 (.vtp) ---
# 웹에서 돌려보기 위해 마지막 시점의 표면 데이터만 추출합니다.
print("Exporting 3D Web Model...")
if not os.path.exists('web_data'):
    os.makedirs('web_data')

# 마지막 타임스텝으로 고정
view.ViewTime = timesteps[-1]
case_foam.UpdatePipeline(timesteps[-1])

# ExtractSurface 필터를 사용하여 3D 볼륨을 가벼운 표면 데이터로 변환 (웹 가시화 최적화)
surf = ExtractSurface(Input=case_foam)
SaveData('web_data/Interactive_Model.vtp', proxy=surf)

print("All tasks completed successfully. 5 Screenshots and 1 3D Web Model generated.")
