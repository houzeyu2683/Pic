import torch
import onnxruntime
import numpy
import os
import random

checkpoint = './log/ramphastidae-2026-0204/getLocation.onnx'
session = onnxruntime.InferenceSession(checkpoint)
collection = []
for _ in range(1):
    # tag = str(number + 1)
    # path = tag.zfill(5)
    
    # section = random.choice(os.listdir("./material/storage/lipread_mp4/"))

    compression = torch.load(
        # f'./material/storage/lipread_pt/{section}/test/{section}_{path}.pt'
        '../Piciformes/material/storage/lipread_pt/ABOUT/test/ABOUT_00001.pt'
    )
    # label = f'{section}_{path}.mp4'
    label = 'ABOUT_00001'
    assert isinstance(compression, torch.Tensor)
    group = []
    for index, item in enumerate(compression):
        vector = item[None, :, :, :].numpy()
        response = session.run(None, {"vector": vector})
        point = response[0].flatten().tolist()
        group += [[label, index] + point]
        continue
    collection += group
    continue
collection = numpy.array(collection)
collection.shape # (*, 5) 分別是 label, index, x, y, z

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

df = pd.DataFrame(collection, columns=['label', 'index', 'x', 'y', 'z'])
df['x'] = df['x'].astype('float')
df['y'] = df['y'].astype('float')
df['z'] = df['z'].astype('float')
df['index'] = df['index'].astype('int')
# df.to_csv('d.csv',index=False)
fig = px.scatter_3d(
    df,
    x='x', y='y', z='z',
    color='label',
    hover_data=['label', 'index']
)
fig.update_traces(marker=dict(size=5))

# 添加箭頭 (線段 + cone) 按 index 順序連接
for lbl, grp in df.groupby('label'):
    grp = grp.sort_values('index')
    x, y, z = grp['x'].values, grp['y'].values, grp['z'].values
    # 連接線段
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=2, color='gray'),
        showlegend=False
    ))
    # 箭頭 (cone) 在每段中點，方向為 i→i+1
    for i in range(len(x) - 1):
        fig.add_trace(go.Cone(
            x=[(x[i] + x[i+1]) / 2],
            y=[(y[i] + y[i+1]) / 2],
            z=[(z[i] + z[i+1]) / 2],
            u=[x[i+1] - x[i]],
            v=[y[i+1] - y[i]],
            w=[z[i+1] - z[i]],
            sizemode='absolute',
            sizeref=0.05,
            showscale=False,
            colorscale=[[0, 'gray'], [1, 'gray']]
        ))

fig.show()
