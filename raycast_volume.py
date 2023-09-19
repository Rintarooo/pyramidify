import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import logging
import os
from typing import Any

from pyramidify import CameraPlotter, CameraMover

# https://yassanabc.com/2021/04/14/%E3%80%90python%E3%80%91logging%E3%81%AE%E6%AD%A3%E3%81%97%E3%81%84%E4%BD%BF%E3%81%84%E6%96%B9/
# https://note.com/enkey/n/na366b382800a
# https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
logger = logging.getLogger(__name__)
# logger = logging.getLogger()
# logging.disable(logging.CRITICAL)

log_file = './out/logging.log'
if os.path.exists(log_file):
    os.remove(log_file)
filehandler = logging.FileHandler(log_file)
# streamhandler = logging.StreamHandler()

# format = '[%(asctime)s][%(levelname)s] %(message)s'
format = '[%(asctime)s][%(levelname)s] {%(module)s.%(funcName)s, l: %(lineno)d} %(message)s'
# datefmt='%Y/%m/%d %I:%M:%S'
datefmt='%I:%M:%Ss'

formatter = logging.Formatter(format, datefmt)
# streamhandler.setFormatter(formatter)
filehandler.setFormatter(formatter)
# logger.addHandler(streamhandler)
logger.addHandler(filehandler)

logger.setLevel(logging.INFO)  # DEBUG #INFO #WARNING

logger.info("Logger INFO mode Start!")
logger.debug("Logger DEBUG mode Start!")
# logger.debug("current dir : {}".format(os.getcwd()))

def degree2radian(deg):
    # return deg * 3.141592659 /180.
    return deg * np.pi / 180.

def normalize(x):
    return x / np.linalg.norm(x)

# def get_camera(fov):
def get_camera(fov, width, height, visualize_scale = 0.5):
    # points = [[ 0,  1,  1, -1, -1], # pyramid verticies # towards -z
    #         [ 0,  1, -1, -1,  1],
    #         [ 0,  -1,  -1,  -1,  -1]]
    # points = [[ 0,  width,  width, -width, -width], # pyramid verticies # towards -z
    #         [ 0,  height, -height, -height, height],
    #         [ 0,  -1.0,  -1.0,  -1.0,  -1.0]]

    # points = [[ 0,  width/2,  width/2, -width/2, -width/2], # pyramid verticies # towards -z
    #         [ 0,  height/2, -height/2, -height/2, height/2],
    #         [ 0,  -1.,  -1.,  -1.,  -1.]]    # points = [[ 0,  1.,1.,-1.,-1.], # pyramid verticies # towards -z
    #         [ 0,  1.,-1.,-1.,1.],
    #         [ 0,  -1.0,  -1.0,  -1.0,  -1.0]]

    ## z negative look at
    # points = [[ 0,  1., 1., -1., -1.], # pyramid verticies # towards -z
    #           [ 0,  1., -1., -1., 1.],
    #           [ 0,  -1., -1., -1., -1.]]    # points = [[ 0,  1.,1.,-1.,-1.], # pyramid verticies # towards -z
    points = [[ 0,  1., 1., -1., -1.], # pyramid verticies # towards z
              [ 0,  1., -1., -1., 1.],
              [ 0,  1., 1., 1., 1.]] 
    # points = [[ 0,  width/2,  width/2, -width/2, -width/2], # pyramid verticies # towards -z
    #         [ 0,  height/2, -height/2, -height/2, height/2],
    #         [ 0,  1.,  1.,  1.,  1.]] 
    lines = [[1, 2], [2, 3], [3, 4], [4, 1], # pyramid lines
            [0, 1], [0, 2], [0, 3], [0, 4]]
    # camera_size = 0.4
    # カメラからスクリーンまでの距離
    # dist_camera2plane = 1. / (2. * np.tan(degree2radian(fov) * 0.5))
    dist_camera2plane = width / (2. * np.tan(degree2radian(fov) * 0.5))
    # print(np.array(points)[0,:])
    # camera = np.array(points) * dist_camera2plane# camera_size
    camera = np.array(points)
    # # print(camera[0,:])

    # camera[2,:] *= dist_camera2plane# camera_size

    # camera *= visualize_scale
    # dist_camera2plane *= visualize_scale
    return camera, points, lines, dist_camera2plane

def raycast(w_, h_, dist_camera2plane, look_dir, camera_right, camera_up, camera_pos):
    # cx = (float)(w_ / 2)
    # cy = (float)(h_ / 2)
    
    # width, height = 3, 3
    # w_ = width * visalize_scale
    
    cx = (float)(w_ / 2)
    cy = (float)(h_ / 2)

    # dw = (float)(2./w_)# 2 = 1.- (-1.)
    # dh = (float)(2./h_)
    # delta = 10**-5
    # lis_w = np.arange(-1., 1.+delta, 2./dw, dtype = float)
    # lis_h = np.arange(-1., 1.+delta, 2./dh, dtype = float)


    ray_dirs = []
    for px in range(w_):
        for py in range(h_):
    # for px in range(-1., 1., dw):
    #     for py in range(-1., 1., dh):
    # for px in lis_w:
    #     for py in lis_h:
            # https://github.com/Rintarooo/Volume-Rendering-Tutorial/blob/f27c64f7909f368dc8205adcef2efa24b40e1e29/Tutorial1/src/VolumeRenderer.cpp#L72-L75
            # Compute the ray direction for this pixel. Same as in standard ray tracing.
            # u_ = -.5 + (px + .5) / (float)(w_-1)
            # v_ =  .5 - (py + .5) / (float)(h_-1)

            u_ = (px + .5) - cx
            v_ = (py + .5) - cy
            # u_ = (px + .5*dw) - cx
            # v_ = (py + .5*dh) - cy

            # logger.debug(f"u_: {u_}, v_: {v_}")
            
            # ray_dir = look_dir * dist_camera2plane + u_ * camera_right + v_ * camera_up
            # ray_dir = np.array([u_ / dist_camera2plane, v_ / dist_camera2plane, -1.])
            ray_dir = np.array([u_ / dist_camera2plane, v_ / dist_camera2plane, 1.])
            ray_dir -= camera_pos
            ray_dirs.append(ray_dir)
    # print(pxl_boarders_x, pxl_boarders_y)
    # grid = np.mgrid[-w_/2:w_/2:1.0, -w_/2:w_/2:1.0]
    # print("grid: ", grid)
    return ray_dirs


def get_trace_camera(camera_pos, camera_up, camera_right, camera_lookat):
    trace1 = {
        "line": {"width": 9}, 
        "mode": "lines", 
        "name": "camera_up", 
        "type": "scatter3d", 
        "x": [camera_pos[0], camera_up[0]], 
        "y": [camera_pos[1], camera_up[1]], 
        "z": [camera_pos[2], camera_up[2]], 
        # "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
        "marker": {"color": "rgb(0, 200, 0)"}, 
        "showlegend": False
    }
    trace2 = {
        "name": "camera_up", 
        "type": "cone", 
        "u": [camera_up[0]-camera_pos[0]], # 矢印の終点のx座標
        "v": [camera_up[1]-camera_pos[1]], 
        "w": [camera_up[2]-camera_pos[2]], 
        "x": [camera_up[0]], # 矢印の始点のx座標
        "y": [camera_up[1]], 
        "z": [camera_up[2]], 
        "sizeref": 0.1, 
        "lighting": {"ambient": 0.8}, 
        "sizemode": "scaled", 
        "hoverinfo": "x+y+z+name", 
        "colorscale": [[0.0, 'rgb(0,200,0)'],[1.0, 'rgb(0,200,0)']],
        "showscale": False, 
        "autocolorscale": False
    }

    trace3 = {
        "line": {"width": 9}, 
        "mode": "lines", 
        "name": "camera_right", 
        "type": "scatter3d", 
        "x": [camera_pos[0], camera_right[0]], 
        "y": [camera_pos[1], camera_right[1]], 
        "z": [camera_pos[2], camera_right[2]], 
        # "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
        # "marker": {"line": {"color": "rgb(200, 0, 0)"}}, 
        "marker": {"color": "rgb(200, 0, 0)"}, 
        "showlegend": False
    }
    trace4 = {
        "name": "camera_right", 
        "type": "cone", 
        "u": [camera_right[0]-camera_pos[0]], # 矢印の終点のx座標
        "v": [camera_right[1]-camera_pos[1]], 
        "w": [camera_right[2]-camera_pos[2]], 
        "x": [camera_right[0]], # 矢印の始点のx座標
        "y": [camera_right[1]], 
        "z": [camera_right[2]], 
        "sizeref": 0.1, 
        "lighting": {"ambient": 0.8}, 
        "sizemode": "scaled", 
        "hoverinfo": "x+y+z+name", 
        "colorscale": [[0.0, 'rgb(200,0,0)'],[1.0, 'rgb(200,0,0)']],
        "showscale": False, 
        "autocolorscale": False
    }

    trace5 = {
        "line": {"width": 9}, 
        "mode": "lines", 
        "name": "camera_lookat", 
        "type": "scatter3d", 
        "x": [camera_pos[0], camera_lookat[0]], 
        "y": [camera_pos[1], camera_lookat[1]], 
        "z": [camera_pos[2], camera_lookat[2]], 
        "marker": {"color": "rgb(0, 0, 200)"}, 
        # "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
        "showlegend": False
    }
    trace6 = {
        "name": "camera_lookat", 
        "type": "cone", 
        "u": [camera_lookat[0]-camera_pos[0]], # 矢印の終点のx座標
        "v": [camera_lookat[1]-camera_pos[1]], 
        "w": [camera_lookat[2]-camera_pos[2]], 
        "x": [camera_lookat[0]], # 矢印の始点のx座標
        "y": [camera_lookat[1]], 
        "z": [camera_lookat[2]], 
        "sizeref": 0.1, 
        "lighting": {"ambient": 0.8}, 
        "sizemode": "scaled", 
        "hoverinfo": "x+y+z+name", 
        "colorscale": [[0.0, 'rgb(0,0,200)'],[1.0, 'rgb(0,0,200)']],
        "showscale": False, 
        "autocolorscale": False
    }
    trace_camera = [trace1, trace2, trace3, trace4, trace5, trace6]
    return trace_camera

def get_trace_ray(camera_pos, ray_dir_0, name):
    trace1 = {
        "line": {"width": 3}, 
        "mode": "lines", 
        "name": name,#"ray_dir_0", 
        "type": "scatter3d", 
        "x": [camera_pos[0], ray_dir_0[0]], 
        "y": [camera_pos[1], ray_dir_0[1]], 
        "z": [camera_pos[2], ray_dir_0[2]], 
        "marker": {"color": "rgb(255, 217, 0)"}, 
        # "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
        "showlegend": False
    }
    trace2 = {
        "name": name, #"ray_dir_0", 
        "type": "cone", 
        "u": [ray_dir_0[0]-camera_pos[0]], # 矢印の終点のx座標
        "v": [ray_dir_0[1]-camera_pos[1]], 
        "w": [ray_dir_0[2]-camera_pos[2]], 
        "x": [ray_dir_0[0]], # 矢印の始点のx座標
        "y": [ray_dir_0[1]], 
        "z": [ray_dir_0[2]], 
        "sizeref": 0.1, 
        "lighting": {"ambient": 0.8}, 
        "sizemode": "scaled", 
        "hoverinfo": "x+y+z+name", 
        "colorscale": [[0.0, 'rgb(255,217,0)'],[1.0, 'rgb(255,217,0)']],
        "showscale": False, 
        "autocolorscale": False
    }
    trace_ray = [trace1, trace2]
    return trace_ray

def get_trace_sampling_point(ray_dir_0_sample):
    trace1 = {
        "name": "sampling_point", 
        "type": "scatter3d", 
        "x": ray_dir_0_sample[:,0], # 点のx座標
        "y": ray_dir_0_sample[:,1], 
        "z": ray_dir_0_sample[:,2], 
        # "sizeref": 0.1, 
        # "lighting": {"ambient": 0.8}, 
        # "sizemode": "scaled", 
        # "hoverinfo": "x+y+z+name", 
        # "colorscale": [[0.0, 'rgb(0,0,0)'],[1.0, 'rgb(0,0,0)']],
        # "showscale": False, 
        # "autocolorscale": False
        "mode": 'markers',
        "marker": {
            "size": 4,
            "color": 'rgb(0,0,0)',
            "colorscale": 'Viridis',
            "opacity": 0.2
        }
    }
    trace_sampling_point = [trace1]
    return trace_sampling_point

def get_trace_screen(camera, lines, camera_pos):
    color='blue'
    name='camera-plane'
    # square draw
    trace_screen_lines = []
    for i, j in lines:
        x, y, z = [[axis[i], axis[j]] for axis in camera]
        # logger.debug(f"lines: [{x},{y},{z}]")
        # x, y, z = [[x, y, z][i] + camera_pos[i] for i in range(3)]
        # logger.debug(f"lines + camera_pos: [{x},{y},{z}]")
        trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line_width=2, line_color=color, name=name)
        trace_screen_lines.append(trace)
        # fig.add_trace(trace)
    # boarder in square draw
    # for i in range(len(pxl_boarders_x)):
    #     x, y, z = [[pxl_boarders_x[i], pxl_boarders_y[j]] for axis in camera]
    # trace_boarder = go.Scatter3d(x=pxl_boarders_x, y=pxl_boarders_y, z=[camera[-1][-1] for _ in range(len(pxl_boarders_x))], mode='lines', line_width=2, line_color=color, name="pixel")
    # grid = np.mgrid[-width/2:(width/2)+0.01:1.0, -width/2:(width/2)+0.01:1.0]
    # print("grid: ", grid)
    # print("grid[0]: ", grid[0])
    # print("grid[1]: ", grid[1])
    # print(grid.shape)
    # print([0.1 for _ in range(grid.shape[1]))])
    # trace_boarder = go.Scatter3d(x=grid[0], y=grid[1], z=[camera[-1][-1] for _ in range(len(grid[0]))], mode='markers+lines', line_width=2, line_color=color, name="pixel")
    # trace_boarder = go.Scatter3d(x=grid[0], y=grid[1], z=[0.1 for _ in range(len(grid[0]))], mode='markers', line_width=2, line_color=color, name="pixel")
    # 2次元グリッドの座標を定義
    # x = [-width/2 + d for d in range(int(width/2))]#[-1.5, 0, 1.5]
    # y = [-width/2 + d for d in range(int(width/2))]#[-1.5, 0, 1.5]


    # dw = (float)(2./width)# 2 = 1.- (-1.)
    # dh = (float)(2./height)
    # delta = 10**-5
    # x = np.arange(-1., 1.+delta, 2./dw, dtype = float)
    # y = np.arange(-1., 1.+delta, 2./dh, dtype = float)
    # print(x)


    # x = [-width/2 + d for d in range(width+1)]#[-1.5, -0.5, 0.5, 1.5]
    # y = [-height/2 + d for d in range(height+1)]#[-1.5, -0.5, 0.5, 1.5]

    # # print(x, y)
    # for i in range(width+1):
    #     for j in range(height+1):
    #         trace_gridx = go.Scatter3d(x=[x[j]]*(width+1), y=y, z=[camera[-1][-1] for _ in range(width+1)], mode='lines', line_width=2, line_color=color, name="pixel")
    #         fig.add_trace(trace_gridx)
    #         trace_gridy = go.Scatter3d(x=x, y=[y[i]]*(height+1), z=[camera[-1][-1] for _ in range(height+1)], mode='lines', line_width=2, line_color=color, name="pixel")
    #         fig.add_trace(trace_gridy)
    # fig.add_trace(trace_boarder)
    x, y, z = camera[:, 1:]
    mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.3, color=color, name=name)
    trace_screen_mesh = [mesh]

    # x, y, z = camera[:, 1:]
    # mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.5, color=color, name=name)
    # fig.add_trace(mesh)
    return trace_screen_lines + trace_screen_mesh

def get_trace_cube(pos_cube, mint, maxt, rgba):
    pos_x, pos_y, pos_z = pos_cube# 3 dimentional tuple
    # create points
    x, y, z = np.meshgrid(
        # np.linspace(mint, maxt, 2), 
        # np.linspace(mint, maxt, 2), 
        # np.linspace(mint, maxt, 2),
        np.linspace(mint+pos_x, maxt+pos_x, 2), 
        np.linspace(mint+pos_y, maxt+pos_y, 2), 
        np.linspace(mint+pos_z, maxt+pos_z, 2),
    )
    # logger.debug(f"plotly cube x: {x}")
    x = x.flatten()
    # logger.debug(f"plotly cube x flatten: {x}")
    y = y.flatten()
    z = z.flatten()
    
    # return go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True, rgba=rgba, lighting={'diffuse': 0.1, 'specular': 2.0, 'roughness': 0.5})
    trace_cube = [go.Mesh3d(x=x, y=y, z=z, alphahull=1.0, color=rgba, name = "cube")]
    return trace_cube

def swap_val(a, b):
    swap_tmp = a
    a = b
    b = swap_tmp
    return a, b
    # return b, a

class AABB():
    def __init__(self, pos_cube_center, mint, maxt, grid_resolution) -> None:
        # self.min_x, self.min_y, self.min_z = -aabb_min, -aabb_min, -aabb_min#-1, -1, -1
        # self.max_x, self.max_y, self.max_z = aabb_max, aabb_max, aabb_max#1, 1, 1
        # self.min_x, self.min_y, self.min_z = -0.5, -0.5, -3.5#-1, -1, -1
        # self.max_x, self.max_y, self.max_z = 0.5, 0.5, -2.5#1, 1, 1

        # self.min_x, self.min_y, self.min_z = -1.0, -1.0, 6.0#-1, -1, -1
        # self.max_x, self.max_y, self.max_z = 1.0, 1.0, 8.0#1, 1, 1


        assert mint < maxt, "range cube"
        self.pos_cube_center = pos_cube_center
        self.pos_x_center, self.pos_y_center, self.pos_z_center = pos_cube_center 
        self.min_x, self.min_y, self.min_z = self.pos_x_center+mint, self.pos_y_center+mint, self.pos_z_center+mint
        self.max_x, self.max_y, self.max_z = self.pos_x_center+maxt, self.pos_y_center+maxt, self.pos_z_center+maxt
        self.min_pos = np.array([self.min_x, self.min_y, self.min_z])
        self.max_pos = np.array([self.max_x, self.max_y, self.max_z])

        # density volume
        self.grid_resolution = grid_resolution
        self.H, self.W, self.D = [self.grid_resolution for _ in range(3)]
        self.num_cell = self.H * self.W * self.D
        # initialize
        self.density_volume = np.zeros((self.H, self.W, self.D, 7))

        
    def ray_intersect_and_get_mint(self, ray_dir, camera_pos):
        # https://github.com/Rintarooo/instant-ngp/blob/090aed613499ac2dbba3c2cede24befa248ece8a/include/neural-graphics-primitives/bounding_box.cuh#L163
        # https://github.com/Southparketeer/SLAM-Large-Dense-Scene-Real-Time-3D-Reconstruction/blob/master/CUDA_Kernel/cu_raycast.cu#L127-L132
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
        # print("ray_dir[0]: ", ray_dir[0])

        # max_ray_scale = 10.0
        # ray_dir *= max_ray_scale

        if_intersect = True
        # print("ray_dir: ", ray_dir)

        # r(t) = o + td, ray_dir is d
        # if ray_dir[0] != 0:# prevent divide by 0
        #     # camera_pos[0] == camera_pos.x
        #     # tmin = aabb.min_x - camera_pos[0] / ray_dir[0]
        #     tmin = (self.min_x - camera_pos[0]) / ray_dir[0]
        #     tmax = (self.max_x - camera_pos[0])/ ray_dir[0]
        # else:
        #     tmin, tmax = 100, 100#0, 0
        if ray_dir[0] == 0:
            tmin = -np.inf
            tmax = np.inf
        else:
            inv_ray_dir_x = 1. / ray_dir[0]
            tmin = (self.min_x - camera_pos[0]) * inv_ray_dir_x
            tmax = (self.max_x - camera_pos[0]) * inv_ray_dir_x

        if (tmin > tmax): tmin, tmax = swap_val(tmin, tmax)
        # print(f"tminx: {tmin}, tmaxx: {tmax}")


        # if ray_dir[1] != 0:
        #     tmin_y = (self.min_y - camera_pos[1])/ ray_dir[1]
        #     tmax_y = (self.max_y - camera_pos[1])/ ray_dir[1]
        # else:
        #     tmin_y, tmax_y = 100, 100
        if ray_dir[1] == 0:
            tmin_y = -np.inf
            tmax_y = np.inf
        else:
            inv_ray_dir_y = 1. / ray_dir[1]
            tmin_y = (self.min_y - camera_pos[1]) * inv_ray_dir_y
            tmax_y = (self.max_y - camera_pos[1]) * inv_ray_dir_y
        if (tmin_y > tmax_y): tmin_y, tmax_y = swap_val(tmin_y, tmax_y)

        if tmin > tmax_y or tmin_y > tmax:
            if_intersect = False
            return tmin, tmax, if_intersect
        # tmin = np.min([tmin, tmin_y])
        # tmax = np.max([tmax, tmax_y])
        if tmin_y > tmin: tmin = tmin_y
        if tmax_y < tmax: tmax = tmax_y
        # print(f"tmin: {tmin}, tmax: {tmax}")



        # if ray_dir[2] != 0:
        #     tmin_z = (self.min_z - camera_pos[2])/ ray_dir[2]
        #     tmax_z = (self.max_z - camera_pos[2])/ ray_dir[2]
        #     print(f"tmin_z: {tmin_z}, tmax_z: {tmax_z}")
        # else:
        #     tmin_z, tmax_z = 100, 100
        inv_ray_dir_z = 1. / ray_dir[2]
        tmin_z = (self.min_z - camera_pos[2]) * inv_ray_dir_z
        tmax_z = (self.max_z - camera_pos[2]) * inv_ray_dir_z
        if (tmin_z > tmax_z): tmin_z, tmax_z = swap_val(tmin_z, tmax_z)
        
        if tmin > tmax_z or tmin_z > tmax:
            if_intersect = False
            return tmin, tmax, if_intersect

        # tmin = np.min([tmin, tmin_z])
        # tmax = np.max([tmax, tmax_z])
        if tmin_z > tmin: tmin = tmin_z
        if tmax_z < tmax: tmax = tmax_z
        # print(f"tmin: {tmin}, tmax: {tmax}")

        return tmin, tmax, if_intersect
    
    def create_density_volume(self, thres_pos, min_density_val, max_density_val, high_density_val, density_mode):
        density = min_density_val
        for i in range(self.H):
            for k in range(self.W):
                for j in range(self.D):
                    # scaling: i is in [0, H-1] --> [0, 1] --> #[0, 2.] --> #[-1., 1.]
                    # x = i/(self.H-1)*2.+(-1.)
                    # y = k/(self.W-1)*2.+(-1.)
                    # z = i/(self.D-1)*2.+(-1.)

                    # scaling: i is in [0, H-1] --> [0, 1] --> #[0, self.max_x-self.min_x] --> #[self.min_x, slf.max_x]
                    x = i/(self.H-1)*(self.max_x-self.min_x)+self.min_x
                    y = k/(self.W-1)*(self.max_y-self.min_y)+self.min_y
                    z = j/(self.D-1)*(self.max_z-self.min_z)+self.min_z
                    density = 0.

                    if density_mode == "sphere":
                        # https://github.com/Rintarooo/Volume-Rendering-Tutorial/blob/f27c64f7909f368dc8205adcef2efa24b40e1e29/Tutorial1/src/main.cpp#L17-L21
                        radius_x = abs(x - self.pos_x_center)
                        radius_y = abs(y - self.pos_y_center)
                        radius_z = abs(z - self.pos_z_center)

                        # Fill the grid with a solid sphere with a very dense inner sphere
                        sphere_radius = radius_x * radius_x + radius_y * radius_y + radius_z * radius_z
                        if sphere_radius < 1.:
                            density = 1.0#sphere_radius
                            if sphere_radius < 0.25:
                                density = high_density_val
                    elif density_mode == "cube":
                        # range_density_val = max_density_val - min_density_val# https://stackoverflow.com/questions/59389241/how-to-generate-a-random-float-in-range-1-1-using-numpy-random-rand
                        # density = np.random.rand()*range_density_val + min_density_val
                        density = min_density_val
                        if self.pos_x_center - thres_pos <= x and x <= self.pos_x_center + thres_pos:
                            if self.pos_y_center - thres_pos <= y and y <= self.pos_y_center + thres_pos:
                                if self.pos_z_center - thres_pos <= z and z <= self.pos_z_center + thres_pos:
                                    # logger.debug(f"high_density_val. because x, y and z is in range min z: {self.pos_z_center - thres_pos} <= z: {z} <= max z: {self.pos_z_center + thres_pos}")
                                    density = high_density_val

                    # white
                    r, g, b = 255, 255, 255

                    # self.density_volume[i][k][j] = (x,y,z,density)
                    
                    # 3d position
                    self.density_volume[i][k][j][0] = x
                    self.density_volume[i][k][j][1] = y
                    self.density_volume[i][k][j][2] = z
                    # rgb
                    self.density_volume[i][k][j][3] = r
                    self.density_volume[i][k][j][4] = g
                    self.density_volume[i][k][j][5] = b
                    # density
                    self.density_volume[i][k][j][6] = density
                    # logger.debug(f"i: {i}, k: {k}, j: {j}")
                    # logger.debug(f"cell pos, x: {x}, y: {y}, z: {z}, density: {density}")

    def read_volume(self, point_pos):
        # nearest neighbor search
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html
        # https://github.com/Rintarooo/code/blob/7f5ea4b995026e9fd1fba127ebce059830e95584/volume-rendering-for-developers/raymarch-chap5.cpp#L247-L259
    #     float evalDensity(const Grid* grid, const Point& p)
    # {
        gridSize_xyz = self.max_pos - self.min_pos
        pLocal = (point_pos - self.min_pos) / gridSize_xyz
        pVoxel = pLocal * self.grid_resolution

        # relative index
        xi =int(np.floor(pVoxel[0]))
        yi =int(np.floor(pVoxel[1]))
        zi =int(np.floor(pVoxel[2]))

        # If outside the grid, then return density zero
        if (xi < 0) or (xi >= self.W): return 0
        if (yi < 0) or (yi >= self.H): return 0
        if (zi < 0) or (zi >= self.D): return 0


        ## nearest neighbor
        # grid_idx = (zi * grid->resolution + yi) * grid->resolution + xi
        # return grid->density[grid_idx]
        x = self.density_volume[xi][yi][zi][0]
        y = self.density_volume[xi][yi][zi][1]
        z = self.density_volume[xi][yi][zi][2]
        density = self.density_volume[xi][yi][zi][6]
        # if point_pos[0] == 0 and point_pos[1]==0:
            # logger.debug(f"\nxi: {xi}, yi: {yi}, zi: {zi}")
        logger.debug(f"\nray point_pos: {point_pos}\ncell pos: [{x}, {y}, {z}]\nread density: {density}")
        return density
    # }

def sampling_light(light_pos, point_pos, delta, aabb):
    # https://github.com/Rintarooo/Volume-Rendering-Tutorial/blob/f27c64f7909f368dc8205adcef2efa24b40e1e29/Tutorial1/src/VolumeRenderer.cpp#L145-L159
    light_dir = point_pos - light_pos
    light_distance = np.linalg.norm(light_dir)
    light_dir = normalize(light_dir)
    
    density_sum = 0.
    logger.debug(f"light_distance: {light_distance}, delta: {delta}, iter_step: {int(light_distance/delta)}")
    
    # tmin, tmax, if_intersect = aabb.ray_intersect_and_get_mint(light_dir, point_pos)
    # if not if_intersect:
    #     # no density on the ray. return transmittance 1.0 (background color rendered)
    #     return 1.
    
    # if if_intersect:
    # for light_step in range(0, int(np.floor(light_distance)), delta):
    for light_step in range(int(light_distance/delta)):
        lighting_point_pos = point_pos + light_step * light_dir
        logger.debug(f"point_pos: {point_pos}, lighting_point_pos: {lighting_point_pos}")
        density_sum += aabb.read_volume(lighting_point_pos)
    
    T_i = np.exp(-density_sum * delta)
    return T_i


# def render_image_from_volume(ray_dirs, w_, h_, num_point, aabb, camera_pos):
def render_image_from_volume(
    ray_dirs: Any,
    w_: Any,
    h_: Any,
    num_point: Any,
    aabb: Any,
    camera_pos: Any,
    light_pos: Any,
) -> None:

    # point_pos = trace_sampling_point_lis[0][0]
    # point_pos = np.array([trace_sampling_point_lis[0][0]["x"][0], trace_sampling_point_lis[0][0]["y"][0], trace_sampling_point_lis[0][0]["z"][0]])
    # cur_density = aabb.read_volume(point_pos)
    
    """"
    absorptionCoef, scatteringCoef = 0.1, 0.1
    # 減衰係数(extinction coefficient) = 吸収係数(absorption coefficient)+散乱係数(scattering coefficient)
    # https://qiita.com/Renard_Renard/items/fd706def17cb8d1e8ec4
    # 減衰が発生し、その結果残る光の割合を透過率(transmittance)と呼ぶ
    # T = e^(-tau)
    # extinctionCoef = 0.1
    extinctionCoef = absorptionCoef + scatteringCoef
    """
    # point_pos = np.array([trace_sampling_point_lis[0][0]["x"][0], trace_sampling_point_lis[0][0]["y"][0], trace_sampling_point_lis[0][0]["z"][0]])
    # cur_density = aabb.read_volume(point_pos)

    
    # for each ray    
    channels = 3
    render_img = np.zeros([w_, h_, channels])
    
    ## for density plot
    density_plot_dic = {"density_lis":[], "tmin":0, "tmax":0, "ray_idx":0}
    density_plot_index_ray = int(len(ray_dirs)/2)#5
    density_plot_dic["ray_idx"] = density_plot_index_ray
    assert density_plot_index_ray < len(ray_dirs), "ray index is over list ray_dirs"

    # for ray_idx in range(len(ray_dirs)):
    cnt_intersect = 0
    for px in range(w_):
        idx_px = px * w_
        for py in range(h_):
            # black
            # render_color = np.array([0,0,0])
            # background_color = np.array([0,0,0])
            render_color = np.array([0.,0.,0.])
            ## light blue
            background_color = np.array([157.,204.,224.])
            ## transmittance, initialize transmission to 1 (fully transparent, so background color is rendered)
            T_ = 1. 
            ray_idx = idx_px + py
            assert ray_idx < len(ray_dirs), "ray index is over list ray_dirs"
            ray_dir = ray_dirs[ray_idx]
            tmin, tmax, if_intersect = aabb.ray_intersect_and_get_mint(ray_dir, camera_pos)
            if not if_intersect:
                tmin = 4.5
                tmax = 9.5
                # black if it does not intersect
                # render_color = np.array([0,0,0])
                render_color = background_color
            assert tmin < tmax, "ray marching range"
            delta = (tmax-tmin)/num_point

            if if_intersect:
                cnt_intersect += 1
                for i in range(num_point):
                    # point_pos = np.array([camera_pos + (tmin + dt*t) * ray_dirs[ray_idx] for t in range(num_point)])
                    point_pos = camera_pos + (tmin+delta*i) * ray_dir
                    cur_density = aabb.read_volume(point_pos)
                    if ray_idx == density_plot_index_ray:
                        density_plot_dic["density_lis"].append(cur_density)
                        if i == 0:
                            density_plot_dic["tmin"] = tmin
                            density_plot_dic["tmax"] = tmax
                    # transmittance *= np.exp(-cur_density * extinctionCoef * delta)
                    alpha_i = 1 - np.exp(-cur_density * delta)
                    T_ *= np.exp(-cur_density * delta)
                    

                    # T_i is the product of 1. - alpha
                    # T_i = sampling_light(light_pos, point_pos, delta, aabb)
                    # T_i = 1.

                    # white, volume cell has homogenious color
                    color = np.array([255,255,255])#np.array([0,0,255])

                    # light emission
                    # weight_i = T_i*alpha_i
                    weight_i = T_*alpha_i

                    logger.debug(f"ray_idx: {ray_idx}, weight_i: {weight_i}, delta: {delta}, ray step: {i}, point_pos: {point_pos}, cur_density: {cur_density}")
                    # render_color += np.uint8(weight_i*color)
                    render_color += weight_i*color
                
            # logger.debug(f"ray_idx: {ray_idx}, px: {px}, py: {py}, np.uint8(render_color): {np.uint8(render_color)}")
            # logger.debug(f"img_x: {w_-px-1}, img_y: {h_-py-1}")
            # render_img[py][px] = np.uint8(render_color)
            logger.debug(f"transmittance: {T_}, background_color: {background_color}, render_color: {render_color}")
            render_color =  T_ * background_color + (1.0-T_) * render_color
            render_img[w_-px-1][h_-py-1] = np.uint8(render_color)
            

    # plt.imsave("./render.png", render_img)
    render_path , density_plot_path = "./out/render.png", "./out/density_plot.png"
    plt.imsave(render_path, np.uint8(render_img))
    logger.info(f"save render_path: {render_path}")
    logger.debug(f"cnt_intersect: {cnt_intersect}/{len(ray_dirs)}")
    
    logger.debug(f"density_lis: {density_plot_dic['density_lis']}")
    plt.plot(np.linspace(density_plot_dic["tmin"], density_plot_dic["tmax"], len(density_plot_dic["density_lis"])), density_plot_dic["density_lis"], ".", markersize=8)
    plt.xlabel(f"tmin:{density_plot_dic['tmin']} - tmax:{density_plot_dic['tmax']}")
    plt.ylabel("density")
    plt.ylim(0, 1)
    plt.title(f"ray index: {density_plot_dic['ray_idx']}")
    logger.info(f"save density_plot_path: {density_plot_path}")
    plt.savefig(density_plot_path)
    # plt.show()

    #     u_, v_ = 1, 1
    #     for i in range(num_point):
    #         # transmittance *= np.exp(-cur_density * extinctionCoef * delta)
    #         alpha_i = 1 - np.exp(-cur_density * delta)
    #         T_i = sampling_light(light_pos, point_pos, delta, aabb, density_vol)
    #         # print("transmittance: ", transmittance)

    #         # white, homogenious color
    #         color = np.array([255,255,255])
    #         weight_i = T_i*alpha_i
    #         render_color += weight_i*color
        
    #     render_img[u_][v_] = np.uint8(render_color)

    # plt.imsave("./render.png", render_img)




def plotly_plot(figshow = True):
    camera_pos = np.array([0, 0, 0])#([1.45, 1.1, 3.1])#([-1, 1, 0])
    camera_lookat = np.array([0, 0, 1])#([0, 0, 0])#([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    look_dir = normalize(camera_lookat - camera_pos)# look direction
    camera_right = np.cross(look_dir, camera_up)
    # camera_up = np.cross(camera_right, look_dir)

    fov = 45#90#150#20#85#10#80
    width, height = 32, 32#256,256#64,64#3, 3#128, 128#1,1#5,5
    camera, points, lines, dist_camera2plane = get_camera(fov, width, height)

    ray_dirs = raycast(width, height, dist_camera2plane, look_dir, camera_right, camera_up, camera_pos)
    
    # M_ext = np.array([[1, 0, 0, 0],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 0, 1]], dtype=float)

    
    trace_camera = get_trace_camera(camera_pos, camera_up, camera_right, camera_lookat)
    trace_screen = get_trace_screen(camera, lines, camera_pos)
  
    pos_cube_center = (0,0,9)#(0,0,7)#(0,0,-3)
    mint, maxt = -2.0, 2.0#-3.0, 3.0#-1.0, 1.0#-0.5, 0.5
    rgba = 'rgba(0,0,0,0.2)'
    trace_cube = get_trace_cube(pos_cube_center, mint, maxt, rgba)
    grid_resolution = 16#128#30
    aabb = AABB(pos_cube_center, mint, maxt, grid_resolution)
    ray_idx_plot_lis = [0, int(len(ray_dirs)/2), int(len(ray_dirs))-1]#[0, int(len(ray_dirs)/4), int(len(ray_dirs)/2), int(len(ray_dirs)*3/4), int(len(ray_dirs))-1]#[0,4,8]

    # ray cast
    tmin_0 = 4.5#3.5
    tmax_0 = 9.5#4.5
    tmin = tmin_0
    tmax = tmax_0
    assert tmin < tmax, "ray marching range"
    trace_ray_lis = []
    trace_sampling_point_lis = []

    num_point = 5#128#30#10
    for ray_idx in ray_idx_plot_lis:
        assert ray_idx <= len(ray_dirs), "ray index should be lower than ray number (render img resolution: H x W)"
        ray_dir = ray_dirs[ray_idx]
        tmin, tmax, if_intersect = aabb.ray_intersect_and_get_mint(ray_dir, camera_pos)
        if not if_intersect:
            tmin = tmin_0
            tmax = tmax_0
        assert tmin < tmax, "ray marching range"
        ray_max = camera_pos + (tmax+3.0) * ray_dir
        # for plotly
        trace_ray = get_trace_ray(camera_pos, ray_max, "ray_dir_"+str(ray_idx))
        trace_ray_lis.append(trace_ray)
        if if_intersect:
            dt = (tmax-tmin)/num_point
            # r(t) = o + td = o + (tmin+dt*i)d
            sampling_point = np.array([camera_pos + (tmin + dt*i) * ray_dirs[ray_idx] for i in range(num_point)])
            trace_sampling_point = get_trace_sampling_point(sampling_point)
            trace_sampling_point_lis.append(trace_sampling_point)

    thres_pos = 0.8#1.0#0.5#0.65
    assert thres_pos > 0, "thres_pos should be positive"
    high_density_val = 0.95
    min_density_val, max_density_val = 0.2, 0.95#0.1, 0.2
    density_mode_lis = ["sphere", "cube"]
    density_mode = density_mode_lis[1]
    aabb.create_density_volume(thres_pos, min_density_val, max_density_val, high_density_val, density_mode)
    

    light_pos = camera_pos#np.array([0,4,9])#np.array([0,0,0])#np.array([0,10,0])
    render_image_from_volume(ray_dirs, width, height, num_point, aabb, camera_pos, light_pos)


    # data = go.Data([trace1, trace2, trace3, trace4, trace5, trace6])
    # data = trace_camera + trace_ray + trace_sampling_point + trace_screen + trace_cube + trace_ray_1 + trace_sampling_point_1 + trace_ray_2 + trace_sampling_point_2
    data = trace_camera + trace_screen + trace_cube# + trace_ray_lis + trace_sampling_point_lis
    for i in range(len(trace_ray_lis)):
        data += trace_ray_lis[i]
    for j in range(len(trace_sampling_point_lis)):
        data += trace_sampling_point_lis[j]
    # fig = Figure(data=data, layout=layout)
    fig = go.Figure(data=data)
    # fig.add_trace(mesh)
    fig.update_layout(showlegend=False)
    # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))
    
    plot_range = 15
    fig.update_layout(
        scene =dict(
            xaxis = dict(range=(-plot_range, plot_range)),
            yaxis = dict(range=(-plot_range, plot_range)),
            zaxis = dict(range=(-plot_range, plot_range)),
    ))
    fig.update_layout(
        scene=dict(
            aspectmode='cube'
        )
    )


    if figshow:
        fig.show()

if __name__=="__main__":
    # https://jun-networks.hatenablog.com/entry/2021/04/02/043216

    figshow = True
    plotly_plot(figshow)

