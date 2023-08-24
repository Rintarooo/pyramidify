import json
import numpy as np
import plotly.graph_objects as go

from pyramidify import CameraPlotter, CameraMover
from scipy.spatial.transform import Rotation

# with open('data/transforms.json', 'r') as file:
with open('mydata/transforms.json', 'r') as file:
    transforms = json.load(file)



plotter = CameraPlotter(transforms)

# poses from transforms.json
for n, frame in enumerate(transforms['frames']):
    M_ext = np.array(frame['transform_matrix'])
    plotter.add_camera(M_ext, color='red', name='frame'+str(n))
    
""" # base_cam.json
base_cam_list = []

for i in [29, 32, 34, 1, 10, 17]:
    mover = CameraMover()
    mover.M_ext = np.array(transforms['frames'][i]['transform_matrix'])
    plotter.add_camera(mover.M_ext.copy(), color='blue', name='camera')
    base_cam_list.append(mover.M_ext)

print(len(base_cam_list), 'cameras added')
 """
plotter.plot_cameras()