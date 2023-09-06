# https://community.plotly.com/t/show-edges-of-the-mesh-in-a-mesh3d-plot/33614/4
import numpy as np
import plotly.graph_objects as go

i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

triangles = np.vstack((i,j,k)).T

x = [0, 0, 1, 1, 0, 0, 1, 1]
y = [0, 1, 1, 0, 0, 1, 1, 0]
z = [0, 0, 0, 0, 1, 1, 1, 1]
vertices = np.vstack((x,y,z)).T
tri_points = vertices[triangles]

#extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
Xe = []
Ye = []
Ze = []
for T in tri_points:
    Xe.extend([T[k%3][0] for k in range(4)]+[ None])
    Ye.extend([T[k%3][1] for k in range(4)]+[ None])
    Ze.extend([T[k%3][2] for k in range(4)]+[ None])
       
#define the trace for triangle sides
lines = go.Scatter3d(
                   x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   name='',
                   line=dict(color= 'rgb(70,70,70)', width=1))  

# data = go.Data([lines])
# fig = go.Figure(data=data)
fig = go.Figure()
fig.add_trace(lines)
fig.show()



# # https://fossies.org/linux/plotly.py/doc/python/3d-mesh.md
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
#         intensity = np.linspace(0, 1, 8, endpoint=True),
#         # i, j and k give the vertices of triangles
#         i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#         j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#         k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
#         name='y',
#         showscale=True
#     )
# ])

# fig.show()