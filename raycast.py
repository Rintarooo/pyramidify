import numpy as np
import plotly.graph_objs as go

def degree2radian(deg):
    # return deg * 3.141592659 /180.
    return deg * np.pi / 180.

def normalize(x):
    return x / np.linalg.norm(x)

def plotly_plot():
    camera_pos = np.array([0, 0, 0])
    camera_lookat = np.array([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    look_dir = normalize(camera_lookat - camera_pos)# look direction
    camera_right = np.cross(look_dir, camera_up)

    fov = 45

    # カメラからスクリーンまでの距離
    dist_camera2plane = 1. / (2. * np.tan(degree2radian(fov) * 0.5))
    print("degree: ", fov)
    print("radian: ", degree2radian(fov))
    print(dist_camera2plane)
    print(look_dir*dist_camera2plane)

    

    # fig = go.Figure(data = go.Cone(
    #     x=camera_pos_x,  # 矢印の始点のx座標
    #     y=camera_pos_y,  # 矢印の始点のy座標
    #     z=camera_pos_z,  # 矢印の始点のz座標
    #     u=[1, 0],  # 矢印の終点のx座標
    #     v=[1, 1],  # 矢印の終点のy座標
    #     w=[1, 1],  # 矢印の終点のz座標
    #     sizemode="absolute",
    #     sizeref=0.2,
    #     anchor="tail"))

    # trace0 = {
    #     "line": {"width": 9}, 
    #     "mode": "lines", 
    #     "name": "camera_right", 
    #     "type": "scatter3d", 
    #     "x": [5.29637232221877, 4.66486936686391], 
    #     "y": [2.93225556101444, 1.4914512476926], 
    #     "z": [1.19745788870817, 2.8650164329552], 
    #     "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
    #     "showlegend": False
    # }
    # trace1 = {
    #     "name": "camera_right", 
    #     "type": "cone", 
    #     "u": [-0.413200740260451], 
    #     "v": [-0.942737328126204], 
    #     "w": [1.09110562202093], 
    #     "x": [4.66486936686391], 
    #     "y": [1.4914512476926], 
    #     "z": [2.8650164329552], 
    #     "sizeref": 0.1, 
    #     "lighting": {"ambient": 0.8}, 
    #     "sizemode": "scaled", 
    #     "hoverinfo": "x+y+z+name", 
    #     "showscale": False, 
    #     "autocolorscale": False
    # }

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

    data = go.Data([trace1, trace2, trace3, trace4, trace5, trace6])
    # fig = Figure(data=data, layout=layout)
    fig = go.Figure(data=data)


    # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    fig.show()

if __name__=="__main__":



    # https://jun-networks.hatenablog.com/entry/2021/04/02/043216
    plotly_plot()

