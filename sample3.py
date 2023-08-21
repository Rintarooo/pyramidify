import plotly.graph_objects as go

# 点の座標を定義
x = [1, 2]
y = [1, 2]
z = [1, 2]

# 3D線分を描画
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])

# 矢印の先端を描画
fig.add_annotation(
    x=x[1],  # 矢印の先端のx座標
    y=y[1],  # 矢印の先端のy座標
    z=z[1],  # 矢印の先端のz座標
    ax=x[0],  # 矢印の基点のx座標
    ay=y[0],  # 矢印の基点のy座標
    az=z[0],  # 矢印の基点のz座標
    xanchor="left",
    yanchor="bottom",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="#636363"
)

fig.show()