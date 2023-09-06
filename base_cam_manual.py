import json
import numpy as np
import plotly.graph_objects as go

from pyramidify_my import CameraPlotter, CameraMover
from scipy.spatial.transform import Rotation




# plotter = CameraPlotter(transforms)
width, height, fov, plot_scale = 3, 3, 40, 1.0
# width, height, fov, plot_scale = 5, 5, 30, 1.0
# width, height, fov, plot_scale = 1, 1, 90, 1.0

use_transform_json = False
transforms = {"empty": ""}
plotter = CameraPlotter(transforms, width, height, fov, plot_scale, use_transform_json)

num_camera = 10
# print(np.cos())
for i in range(num_camera):
    # print(i, np.cos(36*i), 0, np.sin(36*i))
    camera_mover = CameraMover()
    # camera_mover.step_x(5*i)
    radius = 5
    step_x = radius*np.cos(36*i)
    step_z = radius*np.sin(36*i)
    # camera_mover.step_x(step_x)#(5*i)#
    # camera_mover.step_z(step_z)#(5*i)#
    
    camera_mover.step_x(5*i)#
    camera_mover.step_z(5*i)#
    
    # camera_mover.rotate_y(5*i)
    # camera_mover.rotate_x(20*i)
    # camera_mover.rotate_z(20*i)
    M_ext = camera_mover.M_ext

    # M_ext = np.array(frame['transform_matrix'])
    # n = 1
    plotter.add_camera(M_ext, color='blue', name='frame'+str(i))
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
# plotter.plot_camera_rays(scale_plot_ray)
scale_plot_screen = 1#0.01#1#0.01
plotter.plot_screen(scale_plot_screen)