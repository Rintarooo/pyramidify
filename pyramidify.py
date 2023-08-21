import numpy as np
import plotly.graph_objects as go

class CameraPlotter:

    def __init__(self, transforms, camera_size=0.4):
        # `camera["camera_angle_x"]`はカメラのx軸に対する視野角を、
        # `camera["camera_angle_y"]`はカメラのy軸に対する視野角を表しています。
        # fov, represented by radian, not degree.
        self.ca_x = transforms['camera_angle_x']
        self.ca_y = transforms['camera_angle_y']
        self.camera_size = float(camera_size)
        """
        self.points = [[ x_origin,  1,  1, -1, -1], # pyramid verticies
                       [ y_origin,  1, -1, -1,  1],
                       [ z_origin,  1,  1,  1,  1]]
        """

        self.points = [[ 0,  1,  1, -1, -1], # pyramid verticies
                       [ 0,  1, -1, -1,  1],
                       [ 0,  1,  1,  1,  1]]
        self.lines = [[1, 2], [2, 3], [3, 4], [4, 1], # pyramid lines
                      [0, 1], [0, 2], [0, 3], [0, 4]]
        self.camera_list = []

        # self.camera_origin = [0,0,0]
        # self.camera_look = [0,0,1]


    def camera_to_world(self, M_ext):
        """
        input
        M_ext: 4x4 mat
        M_ext convert 3d vector x from world to camera.
        x_c = M_ext * x_w


        return
        R: 3x3 mat
        t: 3x1 vec

        """
        R = M_ext[:3, :3] # rotation matrix
        t = M_ext[:3, 3] # translation vector
        camera = np.array(self.points) * self.camera_size
        camera[0, :] *= np.tan(self.ca_x/2)
        camera[1, :] *= np.tan(self.ca_y/2)
        # print("camera0: ", camera)
        camera = -((R @ camera).T - t).T
        # print("camera1: ", camera)

        # num_points = len(self.points[0])
        # for i in range(num_points):
        #     camera[:,i] = R.T @ camera[:,i] - R.T @ t

        # print(R.T @ camera[:,0] - R.T @ t)
        # print(R.T @ camera[:,1] - R.T @ t)
        # print(R.T @ camera[:,2] - R.T @ t)
        # print(R.T @ camera[:,3] - R.T @ t)
        # print(R.T @ camera[:,4] - R.T @ t)

        # camera = R.T @ camera - R.T @ t
        # np.linalg.inv

        # print(R.T @ camera)
        # print(R.T @ t)
        return camera

    def add_camera(self, M_ext, color, name):
        camera = self.camera_to_world(M_ext)
        self.camera_list.append((camera, color, name))

    def update_camera(self, M_ext, color, name):
        camera = self.camera_to_world(M_ext)
        del self.camera_list[-1]
        self.camera_list.append((camera, color, name))

    def plot_cameras(self):
        fig = go.Figure()
        for n, (camera, color, name) in enumerate(self.camera_list):
            for i, j in self.lines:
                x, y, z = [[axis[i], axis[j]] for axis in camera]
                trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line_width=2, line_color=color, name=name)
                fig.add_trace(trace)
            x, y, z = camera[:, 1:]
            mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.5, color=color, name=name)
            fig.add_trace(mesh)
        fig.update_layout(showlegend=False)
        # https://ai-research-collection.com/plotly-3d-axes/
        # fig.update_layout(
        #     scene = dict(
        #         xaxis = dict(
        #             range=[-1,1],
        #             backgroundcolor="rgb(200, 100, 100)",
        #             gridcolor="white",
        #             showbackground=True,
        #             zerolinecolor="black",
        #         ),
        #         yaxis = dict(
        #             range=[-1,1],
        #             backgroundcolor="rgb(100, 200, 100)",
        #             zerolinecolor="black",
        #         ),
        #         zaxis = dict(
        #             range=[-1,1],
        #             zerolinecolor="black",
        #             ),
        #         # xaxis_title='X AXIS TITLE',
        #         # yaxis_title='Y AXIS TITLE',
        #         # zaxis_title='Z AXIS TITLE',

        #     ),
        # )
        fig.show()

class CameraMover:
    def __init__(self):
        self.M_ext = M_ext = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=float)

    def step_x(self, size):
        self.M_ext[0, 3] += size

    def step_y(self, size):
        self.M_ext[1, 3] += size

    def step_z(self, size):
        self.M_ext[2, 3] += size

    def rotate_x(self, angle):
        angle = np.radians(angle)
        R_theta = np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                            [0, 1, 0, 0],
                            [np.sin(angle), 0, np.cos(angle), 0],
                            [0, 0, 0, 1]], dtype=float)
        self.M_ext = R_theta @ self.M_ext

    def rotate_y(self, angle):
        angle = np.radians(angle)
        R_phi = np.array([[1, 0, 0, 0],
                          [0, np.cos(angle), -np.sin(angle), 0],
                          [0, np.sin(angle), np.cos(angle), 0],
                          [0, 0, 0, 1]], dtype=float)
        self.M_ext = R_phi @ self.M_ext

    def rotate_z(self, angle):
        angle = np.radians(angle)
        R_psi = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                          [np.sin(angle), np.cos(angle), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=float)
        self.M_ext = R_psi @ self.M_ext
