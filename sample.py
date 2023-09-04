import json
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pyramidify import CameraPlotter, CameraMover
from scipy.spatial.transform import Rotation


def debug_plot():
    # fig = go.Figure()
    # fig.show()
    # print("debug_plot")
    
    x = np.linspace(0, 10, 100)
    y = x + np.random.randn(100) 

    # プロット
    plt.plot(x, y, label="test")

    # 凡例の表示
    plt.legend()

    # プロット表示(設定の反映)
    plt.show()

    # fig = go.Figure(
    #     data=[go.Bar(y=[2, 1, 3])],
    #     layout_title_text="A Figure Displayed with fig.show()"
    # )
    # fig.show()


if __name__=="__main__":
    # debug_plot()
    w = 3
    print(2./w)
    delta = 10**-5
    lis = np.arange(-1., 1.+delta, 2./w, dtype = float)
    print(lis)
    # for i in range(-1., 1., 1./w):
        # print(i)

    # transforms = {'camera_angle_x':0.01,'camera_angle_y':0.01} 
    transforms = {'camera_angle_x':0.1,'camera_angle_y':0.1} 
    plotter = CameraPlotter(transforms, camera_size=0.1)

    mover = CameraMover()
    # plotter.add_camera(mover.M_ext.copy(), color='blue', name='camera-0')
    # plotter.plot_cameras()