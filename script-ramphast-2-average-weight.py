import ramphast

device = 'cpu'
model = ramphast.Model(device)
model.activateLayer()

history = './log/ramphast-2026-0204/'
framework = ramphast.Framework(model, device, history)

# # aggregate = framework.Aggregate(history)
checkpoint = [
    '95000.pt',
    '96000.pt',
    '97000.pt',
    '98000.pt',
    '99000.pt',
    '100000.pt'
]
framework.saveWeight(checkpoint)