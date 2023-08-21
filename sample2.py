import plotly.graph_objs as go

# # データを作成します。
# data = go.Scatter(
#     x=[1, 2, 3],
#     y=[1, 2, 3],
#     mode='markers',
# )

# # 矢印のアノテーションを作成します。
# arrow_annotation=dict(
#     ax=2,  # 矢印の始点のx座標
#     ay=2,  # 矢印の始点のy座標
#     az=2,  # 矢印の始点のy座標
#     axref='x',  # x座標の参照
#     ayref='y',  # y座標の参照
#     x=3,  # 矢印の終点のx座標
#     y=3,  # 矢印の終点のy座標
#     z=3,  # 矢印の終点のy座標
#     xref='x',  # x座標の参照
#     yref='y',  # y座標の参照
#     showarrow=True,  # 矢印を表示する
#     arrowhead=1,  # 矢印の形状
#     arrowsize=1,  # 矢印のサイズ
#     arrowwidth=2,  # 矢印の幅
#     arrowcolor='#636363'  # 矢印の色
# )

# layout = go.Layout(
#     title='Plot with Arrow Annotation',
#     annotations=[arrow_annotation]
# )

# fig = go.Figure(data=data, layout=layout)

# fig.show()




# fig = go.Figure(data = go.Cone(
#     x=[1],  # 矢印の始点のx座標
#     y=[1],  # 矢印の始点のy座標
#     z=[1],  # 矢印の始点のz座標
#     u=[1],  # 矢印の終点のx座標
#     v=[1],  # 矢印の終点のy座標
#     w=[1],  # 矢印の終点のz座標
#     sizemode="absolute",
#     sizeref=0.2,
#     anchor="tail"))

# fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
#                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))

# fig.show()




# fig = px.scatter_3d(components, x=0, y=1, z=2)
fig = go.Figure()

# fig.update_layout(
#     scene = dict(
#         annotations=[
#             dict(
#                 # ax=0, ay=0, 
#                 showarrow = True,
#                 arrowsize=arrowsize,
#                 arrowhead=arrowhead,
#                 x = loadings[i, 0]*arrowscale,
#                 y = loadings[i, 1]*arrowscale,
#                 z = loadings[i, 2]*arrowscale,
#                 xanchor="center",
#                 yanchor="bottom",
#                 text = feature,
#                 yshift=5,
#             )
#         for i, feature in enumerate(features)]
#     )
# )

random_x= [0,1]
random_y= [0,1]
random_z= [0,1]
# 3Dスキャッタープロットを作成
fig = go.Figure(data=[go.Scatter3d(
    x=random_x,
    y=random_y,
    z=random_z,
    mode='markers',
    marker=dict(
        size=6,
        color=random_z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
    )]
)

fig.show()