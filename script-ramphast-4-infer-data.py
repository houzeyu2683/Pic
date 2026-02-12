import material
import application

import numpy
import torch
import torchvision.utils
import plotly.express
import plotly.graph_objects
import pandas

interface = {}

version, archive = 'ramphast-v1.0.0', 'getLocation.onnx'
getLocation = application.Service(version, archive)
getLocation.loadSession()

path = './material/storage/lipread_pt/ABOUT/test/ABOUT_00002.pt'
compression = torch.load(path)
group = []
for index, item in enumerate(compression, 1):
    assert torch.is_tensor(item)
    vector = item[None, :, :, :].numpy()
    request = {'vector': vector}
    response = getLocation(request)
    location = response[0]
    location = numpy.array(location).flatten().tolist()
    element = [index] + location
    # point = response[0].flatten().tolist()
    group += [element]
    continue
# collection += group
table = pandas.DataFrame(group, columns=['index', '1', '2', '3'])
table['index'] = table['index'].astype('str')
table['1'] = table['1'].astype('float')
table['2'] = table['2'].astype('float')
table['3'] = table['3'].astype('float')
# df.to_csv('d.csv',index=False)
figure = plotly.express.scatter_3d(
    table,
    x='1', y='2', z='3',
    color='index',
    # hover_data=['label', 'index']
)
figure.update_traces(marker=dict(size=5))

# 添加箭頭 (線段 + cone) 按 index 順序連接
# for lbl, grp in table.groupby('label'):
table = table.sort_values('index')
x, y, z = table['1'].values, table['2'].values, table['3'].values
# 連接線段
figure.add_trace(plotly.graph_objects.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(width=2, color='gray'),
    showlegend=False
))
# 箭頭 (cone) 在每段中點，方向為 i→i+1
for i in range(len(x) - 1):
    figure.add_trace(plotly.graph_objects.Cone(
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
    continue
figure.show()
figure.write_html("display.html")
'''
hub = material.Hub()
batch = hub.getBatch(number=64)
#
group = []
for image in batch['image']:
    image = image[None, :, :, :].numpy()
    #
    request = {'image': image}
    response = getCompression(request)
    compression = response[0]
    #
    request = {"compression": compression}
    response = getReconstruction(request)
    reconstruction = response[0]
    #
    group += [image, reconstruction]
    continue
#
together = numpy.concatenate(group, axis=0)
torchvision.utils.save_image(
    torch.from_numpy(together),
    'result.jpg',
    value_range=(-1, 1), 
    normalize=True
)


'''