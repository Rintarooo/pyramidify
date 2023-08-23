import plotly.graph_objects as go

# 2次元グリッドの座標を定義
x = [-1, 0, 1]
y = [1, 0, -1]

# traceを作成
trace = []

for i in range(3):
    for j in range(3):
        trace.append(go.Scatter(x=x, y=[y[i]]*3,
                                mode='lines',
                                line=dict(color='black')))
        trace.append(go.Scatter(x=[x[j]]*3, y=y,
                                mode='lines',
                                line=dict(color='black')))

# グリッドを描画
fig = go.Figure(data=trace)
fig.show()