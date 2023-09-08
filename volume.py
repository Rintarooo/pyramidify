import numpy as np

def degree2radian(deg):
    return deg * np.pi / 180.

def normalize(x):
    return x / np.linalg.norm(x)

def get_volume(grid_resolution):
    H, W, D = [grid_resolution for _ in range(3)]
    num_celss = H*W*D
    volume_data = np.ones((H, W, D, 7))
    # get_aabb()

    thres = 0.65
    minval, maxval = 0, 0.5
    for i in range(H):
        for k in range(W):
            for j in range(D):
                x = i/(H-1)*2.+(-1.)
                y = k/(W-1)*2.+(-1.)
                z = i/(D-1)*2.+(-1.)
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
                print(f"x: {x}, y: {y}, z: {z}, density: {density}")

    return volume_data

class AABB():
    def __init__(self, aabb_min, aabb_max) -> None:
        self.min_x, self.min_y, self.min_z = -aabb_min, -aabb_min, -aabb_min#-1, -1, -1
        self.max_x, self.max_y, self.max_z = aabb_max, aabb_max, aabb_max#1, 1, 1
        
    # def get_aabb(self): -> None:
    #     return self.min_c

        # tmin = ([-1,-1,-1])
        # tmax = ([1,1,1])

        # min_x, min_y, min_z = -1, -1, -1
        # max_x, max_y, max_z = 1, 1, 1

# def open_dat(path):
#     dat = np.loadtxt(path)
#     print(dat)

def raycast(w_, h_, fl_x, look_dir,camera_pos):
    cx = (float)(w_ / 2)
    cy = (float)(h_ / 2)
    ray_dirs = []
    for px in range(w_):
        for py in range(h_):
            u_ = (px + .5) - cx
            v_ = (py + .5) - cy
            # u_ = (px + .5*dw) - cx
            # v_ = (py + .5*dh) - cy
            # print("u_, v_: ", u_, v_)
            
            # ray_dir = look_dir * dist_camera2plane + u_ * camera_right + v_ * camera_up
            ray_dir = np.array([u_ /fl_x, v_ / fl_x, -1.])
            ray_dir -= camera_pos
            ray_dirs.append(ray_dir)
    return ray_dirs

def ray_intersect_aabb_and_get_mint(ray_dir, camera_pos, aabb):
    # https://github.com/Southparketeer/SLAM-Large-Dense-Scene-Real-Time-3D-Reconstruction/blob/master/CUDA_Kernel/cu_raycast.cu#L127-L132
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-voxel-grids.html
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
    # print("ray_dir[0]: ", ray_dir[0])

    if_intersect = False

    # r(t) = o + td, ray_dir is d
    if ray_dir[0] != 0:# prevent divide by 0
        # camera_pos[0] == camera_pos.x
        tmin = aabb.min_x - camera_pos[0] / ray_dir[0]
        tmax = aabb.max_x - camera_pos[0]/ ray_dir[0]
    else:
        tmin, tmax = 0, 0
    
    if ray_dir[1] != 0:
        tmin_y = aabb.min_y - camera_pos[1]/ ray_dir[1]
        tmax_y = aabb.max_y - camera_pos[1]/ ray_dir[1]
    else:
        tmin_y, tmax_y = 0, 0

    if tmin > tmax_y or tmin_y > tmax: if_intersect = False
    # '''
    # '''
    tmin = np.min([tmin, tmin_y])
    tmax = np.max([tmax, tmax_y])

    if ray_dir[2] != 0:
        tmin_z = aabb.min_y - camera_pos[2]/ ray_dir[2]
        tmax_z = aabb.max_y - camera_pos[2]/ ray_dir[2]
    else:
        tmin_z, tmax_z = 0, 0
    
    if tmin > tmax_z or tmin_z > tmax: if_intersect = False

    tmin = np.min([tmin, tmin_z])
    tmax = np.max([tmax, tmax_z])
    if_intersect = True
    # print("if_intersect: ", if_intersect)
    # print("tmin: ", tmin)
    # print("tmax: ", tmax)



if __name__=="__main__":
    # path = 'dat/christmastree512x499x512.dat'
    # open_dat(path)
    grid_resolution = 6
    volume_data = get_volume(grid_resolution)
    # print(volume_data[0][0][0][-1])

    camera_pos = np.array([0, 0, 0])
    camera_lookat = np.array([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    look_dir = normalize(camera_lookat - camera_pos)# look direction
    camera_right = np.cross(look_dir, camera_up)
    width, height = 3, 3

    fov_x = 45
    # カメラからスクリーンまでの距離
    fl_x = 1. / (2. * np.tan(degree2radian(fov_x) * 0.5))
    ray_dirs = raycast(width, height, fl_x, look_dir,camera_pos)

    # ray cast
    min_t = 2.0
    max_t = 3.0
    ray_dir_0 = camera_pos + (max_t+1.0) * ray_dirs[-1]
    # ray_dir_0_all = [camera_pos + t_ * ray_dirs[-1] for _ in range()]
    ray_dir_0_sample = []

    # min_x, min_y, min_z, max_x, max_y, max_z = get_aabb()

    aabb_min, aabb_max = -1, 1
    aabb = AABB(aabb_min, aabb_max)
    num_point = 20
    dt = (max_t-min_t)/num_point
    for i in range(len(ray_dirs)):
        ray_dir = ray_dirs[i]
        # for t in range(num_point):
        #     d_ = min_t + dt*t
        #     ray_dir = camera_pos + d_ * ray_dir 
        min_t = ray_intersect_aabb_and_get_mint(ray_dir, camera_pos, aabb)

    # num_point = 20
    # dt = (max_t-min_t)/num_point
    # # ray_dir_0_sample = [list(camera_pos + (min_t + dt*t) * ray_dirs[-1]) for t in range(num_point)]
    # ray_dir_0_sample = np.array([camera_pos + (min_t + dt*t) * ray_dirs[-1] for t in range(num_point)])