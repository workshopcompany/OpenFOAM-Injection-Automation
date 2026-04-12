from paraview.simple import *
import os

case_file = "case.foam"
reader = OpenFOAMReader(registrationName='case.foam', FileName=case_file)
reader.MeshRegions = ['internalMesh']
reader.CellArrays = ['alpha.water', 'T', 'p']

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ViewSize = [800, 600]

display = Show(reader, renderView1, 'UnstructuredGridRepresentation')
display.Representation = 'Surface'
ColorBy(display, ('CELLS', 'alpha.water'))

times = reader.TimestepValues
num_steps = 5
step_interval = max(1, len(times) // num_steps)

for i in range(0, len(times), step_interval):
    renderView1.ViewTime = times[i]
    Render()
    SaveScreenshot(f"screenshots/step_{i}.png", renderView1)
