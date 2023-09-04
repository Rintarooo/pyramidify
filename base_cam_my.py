import json
import numpy as np
import plotly.graph_objects as go

from pyramidify_my import CameraPlotter, CameraMover
from scipy.spatial.transform import Rotation

json_path = "mydata/w512h768/transforms.json"
# "mydata/fox/transforms.json"
# "mydata/transforms.json"

with open(json_path, 'r') as file:
    transforms = json.load(file)



# plotter = CameraPlotter(transforms)
width, height, fov, plot_scale = 3, 3, 150, 1.0
# width, height, fov, plot_scale = 5, 5, 30, 1.0
# width, height, fov, plot_scale = 1, 1, 90, 1.0

use_transform_json = True#False
plotter = CameraPlotter(transforms, width, height, fov, plot_scale, use_transform_json)

# poses from transforms.json
for n, frame in enumerate(transforms['frames']):
    if n in [2,4]:
    # if n in [1]:
        print(f"processing {n}/{len(transforms['frames'])} frame ...")
        M_ext = np.array(frame['transform_matrix'])
        plotter.add_camera(M_ext, color='blue', name='frame'+str(n))
        plotter.add_camera_coord(M_ext)
        plotter.add_rays(M_ext)    
""" # base_cam.json
base_cam_list = []

for i in [29, 32, 34, 1, 10, 17]:
    mover = CameraMover()
    mover.M_ext = np.array(transforms['frames'][i]['transform_matrix'])
    plotter.add_camera(mover.M_ext.copy(), color='blue', name='camera')
    base_cam_list.append(mover.M_ext)

print(len(base_cam_list), 'cameras added')
 """

scale_plot_coords = 1#0.01
# plotter.plot_camera_coords(scale_plot_coords)
scale_plot_ray = 1#0.01#1#0.01
plotter.plot_camera_coords(scale_plot_coords, scale_plot_ray)
plotter.plot_camera_rays(scale_plot_ray)
scale_plot_screen = 1#0.01#1#0.01
plotter.plot_screen(scale_plot_screen)