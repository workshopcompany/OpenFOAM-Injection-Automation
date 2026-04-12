from paraview.simple import *

# 1. 데이터 로드 및 뷰 설정
case_foam = PFOAMReader(FileName='case.foam')
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 720]

# 2. 데이터 업데이트
case_foam.UpdatePipeline()

# 3. 초기 가시성 및 컬러링 설정 (속도 U 기준)
display = Show(case_foam, view)
ColorBy(display, ('POINTS', 'U'))
u_lut = GetColorTransferFunction('U')

# 카메라 앵글은 처음 한 번만 맞춥니다
view.ResetCamera()

# 4. 전체 타임스텝 리스트 가져오기
timesteps = case_foam.TimestepValues

if not timesteps:
    print("Error: No timesteps found in case.foam!")
else:
    total_steps = len(timesteps)
    target_count = 5 # 찍고 싶은 사진 개수
    
    # 5. 일정한 간격으로 5개의 인덱스 추출 (시작과 끝 포함)
    if total_steps <= target_count:
        target_indices = list(range(total_steps))
    else:
        # 5등분하여 간격을 계산하는 파이썬 로직
        target_indices = [int(i * (total_steps - 1) / (target_count - 1)) for i in range(target_count)]
        
    print(f"Total timesteps: {total_steps}. Will capture at indices: {target_indices}")

    # 6. 지정된 타임스텝을 순회하며 스크린샷 저장
    for i, idx in enumerate(target_indices):
        t_value = timesteps[idx]
        
        # 해당 시간으로 뷰 이동 및 파이프라인 업데이트
        view.ViewTime = t_value
        case_foam.UpdatePipeline(t_value)
        
        # (선택) 현재 시간의 데이터 범위에 맞춰 컬러바 재조정
        u_lut.RescaleTransferFunctionToDataRange()
        
        # 렌더링
        Render()
        
        # 파일명 지정 (예: result_01_t0.000.png, result_02_t0.050.png)
        # 시간(t_value)이 소수점일 수 있으므로 포맷팅 적용
        filename = f"screenshots/result_{i+1:02d}_t{t_value:.4f}.png"
        SaveScreenshot(filename, view)
        print(f"Saved: {filename} (Simulation Time: {t_value})")

print("All requested screenshots saved successfully.")
