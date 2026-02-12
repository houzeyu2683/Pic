import material
import ramphast

hub = material.Hub()
data = hub.getData(
    number=256
)
validation = hub.getValidation(
    number=64,
    reproducibility=False
)

device = 'cuda'
model = ramphast.Model(device)
model.activateLayer()

history = './log/ramphast-2026-0204' #
framework = ramphast.Framework(model, device, history)

snapshot = 1000
total = -1
accumulation = 8
framework.fitWeight(data, snapshot, total, accumulation, validation)
