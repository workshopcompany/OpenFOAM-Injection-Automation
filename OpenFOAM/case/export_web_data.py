from paraview.simple import *

# 1. 데이터 로드
case_foam = PFOAMReader(FileName='case.foam')
case_foam.UpdatePipeline()

# 2. 마지막 타임스텝 데이터 확보
timesteps = case_foam.TimestepValues
if timesteps:
    t_last = timesteps[-1]
    case_foam.UpdatePipeline(t_last)

# 3. 데이터 경량화 (필요한 경우 표면만 추출)
# 3D 전체 격자는 웹에서 무거울 수 있으므로 ExtractSurface를 주로 사용합니다.
surf = ExtractSurface(Input=case_foam)

# 4. 웹용 데이터로 내보내기 (.vtp 형식)
# 이 파일은 ParaView Glance 웹뷰어에서 바로 열 수 있습니다.
import os
if not os.path.exists('web_data'):
    os.makedirs('web_data')

SaveData('web_data/result_3d.vtp', proxy=surf)
print("3D Web data (VTP) exported successfully.")
