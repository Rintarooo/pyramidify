import numpy as np
import plotly.graph_objs as go

from pyramidify import CameraPlotter, CameraMover

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
    # points = [[ 0,  1., 1., -1., -1.], # pyramid verticies # towards z
    #           [ 0,  1., -1., -1., 1.],
    #           [ 0,  1., 1., 1., 1.]] 
    points = [[ 0,  width/2,  width/2, -width/2, -width/2], # pyramid verticies # towards -z
            [ 0,  height/2, -height/2, -height/2, height/2],
            [ 0,  1.,  1.,  1.,  1.]] 
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
    camera[2,:] *= dist_camera2plane# camera_size

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
            # print("u_, v_: ", u_, v_)
            
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
    # print(ray_dir_0_sample[0,:])
    # print(ray_dir_0_sample)
    # print(ray_dir_0_sample[:,0])
    trace1 = {
        "name": "sampling_point", 
        "type": "scatter3d", 
        # "x": [ray_dir_0_sample[0]], # 矢印の始点のx座標
        # "y": [ray_dir_0_sample[1]], 
        # "z": [ray_dir_0_sample[2]], 
        "x": ray_dir_0_sample[:,0], # 矢印の始点のx座標
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

def get_trace_screen(camera, lines):
    color='blue'
    name='camera-plane'
    # square draw
    trace_screen_lines = []
    for i, j in lines:
        x, y, z = [[axis[i], axis[j]] for axis in camera]
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
    # print(x)
    x = x.flatten()
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
    def __init__(self, pos_cube_center, mint, maxt) -> None:
        # self.min_x, self.min_y, self.min_z = -aabb_min, -aabb_min, -aabb_min#-1, -1, -1
        # self.max_x, self.max_y, self.max_z = aabb_max, aabb_max, aabb_max#1, 1, 1
        # self.min_x, self.min_y, self.min_z = -0.5, -0.5, -3.5#-1, -1, -1
        # self.max_x, self.max_y, self.max_z = 0.5, 0.5, -2.5#1, 1, 1

        # self.min_x, self.min_y, self.min_z = -1.0, -1.0, 6.0#-1, -1, -1
        # self.max_x, self.max_y, self.max_z = 1.0, 1.0, 8.0#1, 1, 1


        assert mint < maxt, "range cube"
        self.pos_cube_center = pos_cube_center
        pos_x, pos_y, pos_z = pos_cube_center 
        self.min_x, self.min_y, self.min_z = pos_x+mint, pos_y+mint, pos_z+mint
        self.max_x, self.max_y, self.max_z = pos_x+maxt, pos_y+maxt, pos_z+maxt
        self.min_pos = np.array([self.min_x, self.min_y, self.min_z])
        self.max_pos = np.array([self.max_x, self.max_y, self.max_z])
        
    def ray_intersect_and_get_mint(self, ray_dir, camera_pos):
        # https://github.com/Rintarooo/instant-ngp/blob/090aed613499ac2dbba3c2cede24befa248ece8a/include/neural-graphics-primitives/bounding_box.cuh#L163
        # https://github.com/Southparketeer/SLAM-Large-Dense-Scene-Real-Time-3D-Reconstruction/blob/master/CUDA_Kernel/cu_raycast.cu#L127-L132
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
        # print("ray_dir[0]: ", ray_dir[0])

        # max_ray_scale = 10.0
        # ray_dir *= max_ray_scale

        if_intersect = True
        print("ray_dir: ", ray_dir)

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
        print(f"tminx: {tmin}, tmaxx: {tmax}")
        
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
        print(f"tmin: {tmin}, tmax: {tmax}")



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
        print(f"tmin: {tmin}, tmax: {tmax}")

        return tmin, tmax, if_intersect
    
    def create_density_volume(self, grid_resolution):
        H, W, D = [grid_resolution for _ in range(3)]
        num_celss = H*W*D
        volume_data = np.ones((H, W, D, 7))
        # get_aabb()

        thres = 0.65
        minval, maxval = 0, 0.5
        for i in range(H):
            for k in range(W):
                for j in range(D):
                    # x = i/(H-1)*2.+(-1.)
                    # y = k/(W-1)*2.+(-1.)
                    # z = i/(D-1)*2.+(-1.)
                    x = i/(H-1)*(self.max_x-self.min_x)+self.min_x#[0,1] --> #[0, self,max_x]
                    y = k/(W-1)*(self.max_y-self.min_y)+self.min_y
                    z = i/(D-1)*(self.max_z-self.min_z)+self.min_z
                    # print(x,y,z)
                    range_size = maxval - minval# https://stackoverflow.com/questions/59389241/how-to-generate-a-random-float-in-range-1-1-using-numpy-random-rand
                    density = np.random.rand()*range_size + minval
                    if 0 < x and x < thres:
                        if 0 < y and y < thres:
                            if 0 < z and z < thres:
                                density = 0.95
                    r, g, b = 1, 1, 1
                    # print(volume_data[i][k][j].shape)
                    # volume_data[i][k][j] = (x,y,z,density)
                    volume_data[i][k][j][0] = x
                    volume_data[i][k][j][1] = y
                    volume_data[i][k][j][2] = z
                    volume_data[i][k][j][3] = r
                    volume_data[i][k][j][4] = g
                    volume_data[i][k][j][5] = b
                    volume_data[i][k][j][6] = density
                    # print(f"x: {x}, y: {y}, z: {z}, density: {density}")
        return volume_data

    def read_volume(self, point_pos, grid_resolution, volume_data):
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html
    #     float evalDensity(const Grid* grid, const Point& p)
    # {
        gridSize = self.max_pos - self.min_pos
        pLocal = (point_pos - self.min_pos) / gridSize
        pVoxel = pLocal * grid_resolution
        xi =int(np.floor(pVoxel[0]))
        yi =int(np.floor(pVoxel[1]))
        zi =int(np.floor(pVoxel[2]))

        ## nearest neighbor
        # grid_idx = (zi * grid->resolution + yi) * grid->resolution + xi
        # return grid->density[grid_idx]
        density = volume_data[xi][yi][zi][6]
        x = volume_data[xi][yi][zi][0]
        y = volume_data[xi][yi][zi][1]
        z = volume_data[xi][yi][zi][2]
        # print("point_pos: ", point_pos)
        # print(f"xi: {xi}, yi: {yi}, zi: {zi}, \nx: {x}, y: {y}, z: {z}, read density: {density}")
        return density
    # }


def plotly_plot():
    camera_pos = np.array([0, 0, 0])
    # camera_lookat = np.array([0, 0, -1])
    camera_lookat = np.array([0, 0, 1])
    camera_up = np.array([0, 1, 0])
    look_dir = normalize(camera_lookat - camera_pos)# look direction
    camera_right = np.cross(look_dir, camera_up)
    # camera_up = np.cross(camera_right, look_dir)

    fov = 45#90#150#45#20#85#10#20#80#45
    width, height = 3,3#5,5#3, 3#1,1#5,5#3, 3
    camera, points, lines, dist_camera2plane = get_camera(fov, width, height)

    ray_dirs = raycast(width, height, dist_camera2plane, look_dir, camera_right, camera_up, camera_pos)
    
    # M_ext = np.array([[1, 0, 0, 0],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 0, 1]], dtype=float)

    # print(ray_dirs)
    
    trace_camera = get_trace_camera(camera_pos, camera_up, camera_right, camera_lookat)
    trace_screen = get_trace_screen(camera, lines)
  
    pos_cube_center = (0,0,9)#(0,0,7)#(0,0,-3)
    mint, maxt = -3.0, 3.0#-1.0, 1.0#-0.5, 0.5
    rgba = 'rgba(0,0,0,0.2)'
    trace_cube =get_trace_cube(pos_cube_center, mint, maxt, rgba)
    # aabb_min, aabb_max = -1, 1
    # aabb = AABB(aabb_min, aabb_max)
    aabb = AABB(pos_cube_center, mint, maxt)
    ray_index_lis = [0, 4, 8]
    # ray cast
    min_t = 4.5#3.5
    max_t = 9.5#4.5
    assert min_t < max_t, "ray marching range"
    trace_ray_lis = []
    trace_sampling_point_lis = []

    num_point = 10
    for ray_idx in ray_index_lis:
        assert ray_idx <= len(ray_dirs), "ray index should lower than ray number"
        ray_dir = ray_dirs[ray_idx]
        tmin, tmax, if_intersect = aabb.ray_intersect_and_get_mint(ray_dir, camera_pos)
        # print("if_intersect: ", if_intersect)
        # print("tmin: ", tmin)
        # print("tmax: ", tmax)
        if not if_intersect:
            tmin = 4.5
            tmax = 9.5
        assert min_t < max_t, "ray marching range"
        dt = (tmax-tmin)/num_point
        # r(t) = o + td
        sampling_point = np.array([camera_pos + (tmin + dt*t) * ray_dirs[ray_idx] for t in range(num_point)])
        ray_max = camera_pos + (max_t+3.0) * ray_dir

        # for plotly
        trace_ray = get_trace_ray(camera_pos, ray_max, "ray_dir_"+str(ray_idx))
        trace_sampling_point = get_trace_sampling_point(sampling_point)
        trace_ray_lis.append(trace_ray)
        trace_sampling_point_lis.append(trace_sampling_point)

    grid_resolution =3
    density_vol = aabb.create_density_volume(grid_resolution)
    # point_pos = trace_sampling_point_lis[0][0]
    point_pos = np.array([trace_sampling_point_lis[0][0]["x"][0], trace_sampling_point_lis[0][0]["y"][0], trace_sampling_point_lis[0][0]["z"][0]])
    density = aabb.read_volume(point_pos, grid_resolution, density_vol)


    # data = go.Data([trace1, trace2, trace3, trace4, trace5, trace6])
    # data = trace_camera + trace_ray + trace_sampling_point + trace_screen + trace_cube + trace_ray_1 + trace_sampling_point_1 + trace_ray_2 + trace_sampling_point_2
    data = trace_camera + trace_screen + trace_cube# + trace_ray_lis + trace_sampling_point_lis
    for k in range(len(trace_ray_lis)):
        data += trace_ray_lis[k] + trace_sampling_point_lis[k]
    # fig = Figure(data=data, layout=layout)
    fig = go.Figure(data=data)
    # fig.add_trace(mesh)
    fig.update_layout(showlegend=False)
    # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    fig.show()

if __name__=="__main__":
    # https://jun-networks.hatenablog.com/entry/2021/04/02/043216
    plotly_plot()

