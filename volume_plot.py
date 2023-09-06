import plotly.graph_objects as go
import numpy as np

# X, Y, Z = np.mgrid[0:2:3j, 0:2:3j, 0:2:3j]
# print(X)

# キューブの頂点
x = [0, 0, 1, 1, 0, 0, 1, 1]
y = [0, 1, 1, 0, 0, 1, 1, 0]
z = [0, 0, 0, 0, 1, 1, 1, 1]

# # キューブのエッジ
# i = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
# j = [1, 2, 3, 4, 5, 0, 3, 6, 2, 7, 5, 0, 6, 1, 7, 2, 3, 4]
# k = [5, 4, 7, 1, 0, 2, 0, 3, 1, 2, 1, 5, 1, 4, 2, 5, 6, 5]
# i, j and k give the vertices of triangles
i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],

# 3Dキューブを描画
trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=1.0)

fig = go.Figure(data=[trace])
fig.show()

# import plotly.graph_objects as go
# import numpy as np

# fig = go.Figure(data=[
#     go.Mesh3d(
#         # 8 vertices of a cube
#         x=[0, 0, 1, 1, 0, 0, 1, 1],
#         y=[0, 1, 1, 0, 0, 1, 1, 0],
#         z=[0, 0, 0, 0, 1, 1, 1, 1],
#         colorbar_title='z',
#         colorscale=[[0, 'gold'],
#                     [0.5, 'mediumturquoise'],
#                     [1, 'magenta']],
#         # Intensity of each vertex, which will be interpolated and color-coded
#         intensity = np.linspace(0, 1, 12, endpoint=True),
#         intensitymode='cell',
#         # i, j and k give the vertices of triangles
#         i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#         j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#         k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
#         name='y',
#         showscale=True
#     )
# ])

# fig.show()


import plotly.graph_objects as go
import numpy as np

def cubes(size, pos_x, pos_y, pos_z, color):
    # create points
    x, y, z = np.meshgrid(
        np.linspace(pos_x-size/2, pos_x+size/2, 2), 
        np.linspace(pos_y-size/2, pos_y+size/2, 2), 
        np.linspace(pos_z-size/2, pos_z+size/2, 2),
    )
    print(x)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    # return go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True, color=color, lighting={'diffuse': 0.1, 'specular': 2.0, 'roughness': 0.5})
    return go.Mesh3d(x=x, y=y, z=z, alphahull=0.8, color=color)
    # return go.Mesh3d(x=x, y=y, z=z, opacity=1, color=color)


def cubes_minmax(pos_cube, mint, maxt, color):
    pos_x, pos_y, pos_z = pos_cube
    # create points
    x, y, z = np.meshgrid(
        # np.linspace(mint, maxt, 2), 
        # np.linspace(mint, maxt, 2), 
        # np.linspace(mint, maxt, 2),
        np.linspace(mint+pos_x, maxt+pos_x, 2), 
        np.linspace(mint+pos_y, maxt+pos_y, 2), 
        np.linspace(mint+pos_z, maxt+pos_z, 2),
    )
    print(x)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    # return go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True, color=color, lighting={'diffuse': 0.1, 'specular': 2.0, 'roughness': 0.5})
    return go.Mesh3d(x=x, y=y, z=z, alphahull=0.8, color=color)
    # return go.Mesh3d(x=x, y=y, z=z, opacity=1, color=color)

if __name__=="__main__":
    fig = go.Figure()
    # set edge length of cubes
    size = 1
    pos_cube = (0,0,5)

    # add outer cube
    # fig.add_trace(cubes(size,0,0,0, 'rgba(0,0,0,0.4)'))
    fig.add_trace(cubes_minmax(pos_cube, -1, 1, 'rgba(0,0,0,0.2)'))
    fig.show()




# import numpy as np
# import plotly.graph_objects as go
# fig = go.Figure(data=[
#     go.Mesh3d(
#         # 8 vertices of a cube
#         x=[0, 0, 1, 1, 0, 0, 1, 1],
#         y=[0, 1, 1, 0, 0, 1, 1, 0],
#         z=[0, 0, 0, 0, 1, 1, 1, 1],
#         colorbar_title='z',
#         colorscale=[[0, 'gold'],
#                     [0.5, 'mediumturquoise'],
#                     [1, 'magenta']],
#         # Intensity of each vertex, which will be interpolated and color-coded
#         intensity = np.linspace(0, 1, 12, endpoint=True),
#         intensitymode='cell',
#         # i, j and k give the vertices of triangles
#         i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#         j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#         k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
#         name='y',
#         showscale=True
#     )
# ])

# fig.show()