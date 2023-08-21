import numpy as np
import plotly.graph_objs as go

def degree2radian(deg):
    # return deg * 3.141592659 /180.
    return deg * np.pi / 180.

def plotly_plot():

    # uvz = np.array([1,1,1])
    
    uvz = [[1],[1],[1]]
    xyz = [[0],[0],[0]]
    # fig = go.Figure(data = go.Cone(
    #     x=xyz[0],  # 矢印の始点のx座標
    #     y=xyz[1],  # 矢印の始点のy座標
    #     z=xyz[2],  # 矢印の始点のz座標
    #     u=uvz[0],  # 矢印の終点のx座標
    #     v=uvz[1],  # 矢印の終点のy座標
    #     w=uvz[2],  # 矢印の終点のz座標
    #     sizemode="absolute",
    #     sizeref=0.2,
    #     anchor="tail"))

    # fig = go.Figure(data = go.Cone(
    #     x=[1, 2],  # 矢印の始点のx座標
    #     y=[1, 2],  # 矢印の始点のy座標
    #     z=[1, 2],  # 矢印の始点のz座標
    #     u=[1, 0],  # 矢印の終点のx座標
    #     v=[1, 1],  # 矢印の終点のy座標
    #     w=[1, 1],  # 矢印の終点のz座標
    #     sizemode="absolute",
    #     sizeref=0.2,
    #     anchor="tail"))

    camera_pos = np.array([0, 0, 1])
    camera_lookat = np.array([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    look_dir = camera_lookat - camera_pos
    camera_right = np.cross(look_dir, camera_up)

    camera_pos_x = [camera_pos[0] for _ in range(3)]
    camera_pos_y = [camera_pos[1] for _ in range(3)]
    camera_pos_z = [camera_pos[2] for _ in range(3)]
    

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

    trace0 = {
        "line": {"width": 9}, 
        "mode": "lines", 
        "name": "PC2", 
        "type": "scatter3d", 
        "x": [5.29637232221877, 4.66486936686391], 
        "y": [2.93225556101444, 1.4914512476926], 
        "z": [1.19745788870817, 2.8650164329552], 
        "marker": {"line": {"color": "rgb(35, 155, 118)"}}, 
        "showlegend": False
    }
    trace1 = {
        "name": "PC2", 
        "type": "cone", 
        "u": [-0.413200740260451], 
        "v": [-0.942737328126204], 
        "w": [1.09110562202093], 
        "x": [4.66486936686391], 
        "y": [1.4914512476926], 
        "z": [2.8650164329552], 
        "sizeref": 0.1, 
        "lighting": {"ambient": 0.8}, 
        "sizemode": "scaled", 
        "hoverinfo": "x+y+z+name", 
        "showscale": False, 
        "autocolorscale": False
    }

    data = go.Data([trace0, trace1])
    # fig = Figure(data=data, layout=layout)
    fig = go.Figure(data=data)


    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                 camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    fig.show()

if __name__=="__main__":

    # camera_pos = (1, 1, 1)
    # camera_lookat = (0, 0, 0)
    # camera_up = (0, 1, 0)
    camera_pos = np.array([0, 0, 1])
    camera_lookat = np.array([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    look_dir = camera_lookat - camera_pos
    camera_right = np.cross(look_dir, camera_up)

    fov = 45

    # カメラからスクリーンまでの距離
    dist_camera2plane = 1. / (2. * np.tan(degree2radian(fov) * 0.5))
    print(dist_camera2plane)

    # https://jun-networks.hatenablog.com/entry/2021/04/02/043216
    plotly_plot()
