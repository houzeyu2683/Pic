import ramphast
import application
import material

device = 'cuda'
model = ramphast.Model(device)
model.activateLayer()

path = './log/ramphast-2026-0204/weight.pt'
model.loadCheckpoint(path)

hub = material.Hub()
number = 1
batch = hub.getBatch(number)
vector = batch['vector']
location = model.getLocation(vector)
#
luggage = application.Luggage(folder='./log/ramphast-2026-0204')
#
data = [vector]
key = ['vector']
luggage.exportModule(
    model=model, 
    method='getLocation', 
    data=data, 
    key=key, 
    archive='getLocation.onnx'
)