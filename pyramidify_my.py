import numpy as np
import plotly.graph_objects as go

def degree2radian(deg):
    return deg * np.pi / 180.

def normalize(x):
    return x / np.linalg.norm(x)


class CameraPlotter:

    def __init__(self, transforms, width, height, fov, plot_scale = 0.1):
        # `camera["camera_angle_x"]`はカメラのx軸に対する視野角を、
        # `camera["camera_angle_y"]`はカメラのy軸に対する視野角を表しています。
        # fov, represented by radian, not degree.
        self.ca_x = transforms['camera_angle_x']
        self.ca_y = transforms['camera_angle_y']
        # self.camera_size = float(camera_size)
        """
        self.points = [[ x_origin,  1,  1, -1, -1], # pyramid verticies
                       [ y_origin,  1, -1, -1,  1],
                       [ z_origin,  1,  1,  1,  1]]
        """

        # self.points = [[ 0,  1,  1, -1, -1], # pyramid verticies
        #                [ 0,  1, -1, -1,  1],
        #                [ 0,  1,  1,  1,  1]]
        self.points = [[ 0,  width/2,  width/2, -width/2, -width/2], 
                       [ 0,  height/2, -height/2, -height/2, height/2],
                       [ 0,  1.0,  1.0,  1.0,  1.0]]# pyramid verticies # towards -z
        self.plot_scale = plot_scale
        # self.points *= self.plot_scale
        
        # self.points = [[ 0,  1,  1, -1, -1], # pyramid verticies
        #                [ 0,  1, -1, -1,  1],
        #                [ 0,  -1,  -1,  -1,  -1]]
        self.lines = [[1, 2], [2, 3], [3, 4], [4, 1], # pyramid lines
                      [0, 1], [0, 2], [0, 3], [0, 4]]
        self.camera_list = []

        # カメラからスクリーンまでの距離
        self.fl_x = width / (2. * np.tan(degree2radian(fov) * 0.5))
        self.fl_y = height / (2. * np.tan(degree2radian(fov) * 0.5))

        self.cx = (float)(width / 2)
        self.cy = (float)(height / 2)
        self.ray_dirs_list = []


        self.camera_pos = np.array([0, 0, 0])
        self.camera_lookat = np.array([0, 0, 1])
        self.camera_up = np.array([0, 1, 0])
        # self.camera_up = np.array([0, -1, 0])
        self.look_dir = normalize(self.camera_lookat - self.camera_pos)# look direction
        self.camera_right = np.cross(self.look_dir, self.camera_up)
        self.camera_pos_world_list = []
        self.camera_up_world_list = []
        self.camera_lookat_world_list = []
        self.camera_right_world_list = []

        # self.camera_origin = [0,0,0]
        # self.camera_look = [0,0,1]
        self.fig = go.Figure()



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

        # camera = np.array(self.points) * self.camera_size
        # camera[0, :] *= np.tan(self.ca_x/2)
        # camera[1, :] *= np.tan(self.ca_y/2)

        camera = np.array(self.points)
        camera *= self.plot_scale
        # # print(camera[0,:])
        camera[2,:] *= self.fl_x



        # print("camera0: ", camera)

        camera = -((R @ camera).T - t).T
        
        # print("camera1: ", camera)

        # num_points = len(self.points[0])
        # for i in range(num_points):
        #     print("camera[:,i]: ", camera[:,i])
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

    # def plot_cameras(self):
    #     fig = go.Figure()
    #     for n, (camera, color, name) in enumerate(self.camera_list):
    #         for i, j in self.lines:
    #             x, y, z = [[axis[i], axis[j]] for axis in camera]
    #             trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line_width=2, line_color=color, name=name)
    #             fig.add_trace(trace)
    #         x, y, z = camera[:, 1:]
    #         mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.3, color=color, name=name)
    #         fig.add_trace(mesh)
    #     fig.update_layout(showlegend=False)
    #     fig.show()

    def plot_cameras(self):
        # fig = go.Figure()
        for n, (camera, color, name) in enumerate(self.camera_list):
            for i, j in self.lines:
                x, y, z = [[axis[i], axis[j]] for axis in camera]
                trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line_width=2, line_color=color, name=name)
                self.fig.add_trace(trace)
            x, y, z = camera[:, 1:]
            mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.3, color=color, name=name)
            self.fig.add_trace(mesh)
        self.fig.update_layout(showlegend=False)
        self.fig.show()
    
    # def raycast(self):
    #     for px in range(w_):
    #         for py in range(h_):
    #             # https://github.com/Rintarooo/Volume-Rendering-Tutorial/blob/f27c64f7909f368dc8205adcef2efa24b40e1e29/Tutorial1/src/VolumeRenderer.cpp#L72-L75
    #             # Compute the ray direction for this pixel. Same as in standard ray tracing.
    #             # u_ = -.5 + (px + .5) / (float)(w_-1)
    #             # v_ =  .5 - (py + .5) / (float)(h_-1)

    #             u_ = (px + .5) - cx
    #             v_ = (py + .5) - cy
    #             # pxl_boarders_x.append(px-cx)
    #             # pxl_boarders_y.append(py-cy)
    #             # print("u_, v_: ", u_, v_)
    #             ray_dir = look_dir * self.fl_x + u_ * camera_right + v_ * camera_up
    #             self.ray_dirs.append(ray_dir)
    #             # return ray_dirs

    # def add_rays(self, M_ext, color, name):

    def add_camera_coord(self, M_ext):
        R = M_ext[:3, :3] # rotation matrix
        t = M_ext[:3, 3] # translation vector

        # self.camera_pos = np.array([0, 0, 0])
        # self.camera_lookat = np.array([0, 0, 1])
        # self.camera_up = np.array([0, 1, 0])
        # self.look_dir = normalize(self.camera_lookat - self.camera_pos)# look direction
        # self.camera_right = np.cross(self.look_dir, self.camera_up)

        camera_pos = -((R @ self.camera_pos).T - t).T
        camera_lookat = -((R @ self.camera_lookat).T - t).T
        camera_up = -((R @ self.camera_up).T - t).T
        camera_right = -((R @ self.camera_right).T - t).T
        self.camera_pos_world_list.append(camera_pos)
        self.camera_up_world_list.append(camera_up)
        self.camera_lookat_world_list.append(camera_lookat)
        self.camera_right_world_list.append(camera_right)

    def plot_camera_coords(self, plot_scale=0.1):
        for i in range(len(self.camera_pos_world_list)):
            camera_pos = self.camera_pos_world_list[i]
            camera_up = self.camera_up_world_list[i]
            camera_lookat = self.camera_lookat_world_list[i]
            camera_right = self.camera_right_world_list[i]

            trace1 = {
                "line": {"width": 4}, 
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
                "line": {"width": 4}, 
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
                "line": {"width": 4}, 
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
            self.fig.add_traces([trace1, trace2, trace3, trace4, trace5, trace6])
            # self.fig.add_traces
    


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
