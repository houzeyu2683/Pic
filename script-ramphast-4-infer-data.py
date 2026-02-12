import material
import application

import numpy
import torch
import torchvision.utils

interface = {}

version, archive = 'ramphast-v1.0.0', 'getCompression.onnx'
getCompression = application.Service(version, archive)
getCompression.loadSession()

version, archive = 'ramphast-v1.0.0', 'getReconstruction.onnx'
getReconstruction = application.Service(version, archive)
getReconstruction.loadSession()
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