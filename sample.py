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
    transforms = {'camera_angle_x':0.01,'camera_angle_y':0.01} 
    plotter = CameraPlotter(transforms)

    mover = CameraMover()
    plotter.add_camera(mover.M_ext.copy(), color='blue', name='camera')
    plotter.plot_cameras()