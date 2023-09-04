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

    points = [[ 0,  width/2,  width/2, -width/2, -width/2], # pyramid verticies # towards -z
            [ 0,  height/2, -height/2, -height/2, height/2],
            [ 0,  -1.,  -1.,  -1.,  -1.]]    # points = [[ 0,  1.,1.,-1.,-1.], # pyramid verticies # towards -z
    #         [ 0,  1.,-1.,-1.,1.],
    #         [ 0,  -1.0,  -1.0,  -1.0,  -1.0]]
    # points = [[ 0,  1., 1., -1., -1.], # pyramid verticies # towards -z
    #           [ 0,  1., -1., -1., 1.],
    #           [ 0,  -1., -1., -1., -1.]]    # points = [[ 0,  1.,1.,-1.,-1.], # pyramid verticies # towards -z
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

    camera *= visualize_scale
    dist_camera2plane *= visualize_scale

    return camera, points, lines, dist_camera2plane

def raycast(width, height, dist_camera2plane, look_dir, camera_right, camera_up):
    # cx = (float)(w_ / 2)
    # cy = (float)(h_ / 2)
    
    # width, height = 3, 3
    # w_ = width * visalize_scale
    
    cx = (float)(w_ / 2)
    cy = (float)(h_ / 2)

    dw = (float)(2./w_)# 2 = 1.- (-1.)
    dh = (float)(2./h_)
    delta = 10**-5
    lis_w = np.arange(-1., 1.+delta, 2./dw, dtype = float)
    lis_h = np.arange(-1., 1.+delta, 2./dh, dtype = float)


    ray_dirs = []
    # for px in range(w_):
    #     for py in range(h_):
    # for px in range(-1., 1., dw):
    #     for py in range(-1., 1., dh):
    for px in lis_w:
        for py in lis_h:
            # https://github.com/Rintarooo/Volume-Rendering-Tutorial/blob/f27c64f7909f368dc8205adcef2efa24b40e1e29/Tutorial1/src/VolumeRenderer.cpp#L72-L75
            # Compute the ray direction for this pixel. Same as in standard ray tracing.
            # u_ = -.5 + (px + .5) / (float)(w_-1)
            # v_ =  .5 - (py + .5) / (float)(h_-1)

            # u_ = (px + .5) - cx
            # v_ = (py + .5) - cy
            u_ = (px + .5*dw) - cx
            v_ = (py + .5*dh) - cy
            # print("u_, v_: ", u_, v_)
            ray_dir = look_dir * dist_camera2plane + u_ * camera_right + v_ * camera_up
            ray_dirs.append(ray_dir)
    # print(pxl_boarders_x, pxl_boarders_y)
    # grid = np.mgrid[-w_/2:w_/2:1.0, -w_/2:w_/2:1.0]
    # print("grid: ", grid)
    return ray_dirs



def plotly_plot():
    camera_pos = np.array([0, 0, 0])
    camera_lookat = np.array([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    look_dir = normalize(camera_lookat - camera_pos)# look direction
    camera_right = np.cross(look_dir, camera_up)

    fov = 45#90#150#45#20#85#10#20#80#45

    # カメラからスクリーンまでの距離
    # dist_camera2plane = 1. / (2. * np.tan(degree2radian(fov) * 0.5))
    # print("degree: ", fov)
    # print("radian: ", degree2radian(fov))

    # print("dist_camera2plane: ", dist_camera2plane)
    # print(look_dir*dist_camera2plane)

    width, height = 3,3#5,5#3, 3#1,1#5,5#3, 3
    # camera, points, lines = get_camera(fov, width, height)
    camera, points, lines, dist_camera2plane = get_camera(fov, width, height)

    ray_dirs = raycast(width, height, dist_camera2plane, look_dir, camera_right, camera_up)
    
    # print(ray_dirs)

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

    # ray cast
    min_t = 2.0
    max_t = 3.0
    ray_dir_0 = camera_pos + (max_t+1.0) * ray_dirs[-1]
    # ray_dir_0_all = [camera_pos + t_ * ray_dirs[-1] for _ in range()]
    ray_dir_0_sample = []
    num_point = 20
    dt = (max_t-min_t)/num_point
    # ray_dir_0_sample = [list(camera_pos + (min_t + dt*t) * ray_dirs[-1]) for t in range(num_point)]
    ray_dir_0_sample = np.array([camera_pos + (min_t + dt*t) * ray_dirs[-1] for t in range(num_point)])
    # for i in range(t_):
    #     ray_dirs
    trace7 = {
        "line": {"width": 3}, 
        "mode": "lines", 
        "name": "ray_dir_0", 
        "type": "scatter3d", 
        "x": [camera_pos[0], ray_dir_0[0]], 
        "y": [camera_pos[1], ray_dir_0[1]], 
        "z": [camera_pos[2], ray_dir_0[2]], 
        "marker": {"color": "rgb(255, 217, 0)"}, 
        # "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
        "showlegend": False
    }
    trace8 = {
        "name": "ray_dir_0", 
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

    # print(ray_dir_0_sample[0,:])
    # print(ray_dir_0_sample)
    # print(ray_dir_0_sample[:,0])
    trace9 = {
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

    # data = go.Data([trace1, trace2, trace3, trace4, trace5, trace6])
    # data = go.Data([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8])
    data = go.Data([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9])

    # fig = Figure(data=data, layout=layout)
    fig = go.Figure(data=data)

    color='blue'
    name='camera-plane'
    # square draw
    for i, j in lines:
        x, y, z = [[axis[i], axis[j]] for axis in camera]
        trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line_width=2, line_color=color, name=name)
        fig.add_trace(trace)
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


    x = [-width/2 + d for d in range(width+1)]#[-1.5, -0.5, 0.5, 1.5]
    y = [-height/2 + d for d in range(height+1)]#[-1.5, -0.5, 0.5, 1.5]
    # print(x, y)
    for i in range(width+1):
        for j in range(height+1):
            trace_gridx = go.Scatter3d(x=[x[j]]*(width+1), y=y, z=[camera[-1][-1] for _ in range(width+1)], mode='lines', line_width=2, line_color=color, name="pixel")
            fig.add_trace(trace_gridx)
            trace_gridy = go.Scatter3d(x=x, y=[y[i]]*(height+1), z=[camera[-1][-1] for _ in range(height+1)], mode='lines', line_width=2, line_color=color, name="pixel")
            fig.add_trace(trace_gridy)
    # fig.add_trace(trace_boarder)
    x, y, z = camera[:, 1:]
    mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.3, color=color, name=name)
    fig.add_trace(mesh)
    fig.update_layout(showlegend=False)

    # x, y, z = camera[:, 1:]
    # mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.5, color=color, name=name)
    # fig.add_trace(mesh)



    # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    fig.show()

if __name__=="__main__":
    # https://jun-networks.hatenablog.com/entry/2021/04/02/043216
    plotly_plot()

