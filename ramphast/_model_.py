import torch
import diffusers
import tensordict
import diffusers.models.resnet
import diffusers.models
import diffusers.models.unets.unet_2d_blocks
import safetensors.torch
import diffusers.models.embeddings
import diffusers.models.downsampling

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def loadCheckpoint(self, path: str) -> bool:
        state_dict = safetensors.torch.load_file(path)
        self.load_state_dict(state_dict)
        return(True)

    def activateLayer(self) -> bool:
        # block = diffusers.models.unets.unet_2d_blocks
        block = diffusers.models.unets.unet_2d_blocks.DownEncoderBlock2D
        layer = {
            '(1) projection': block(
                in_channels=32, 
                out_channels=64
            ),
            '(2) projection': block(
                in_channels=64, 
                out_channels=128
            ),
            '(3) projection': block(
                in_channels=128, 
                out_channels=256
            ),
            'location': torch.nn.Linear(256, 3)
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getLocation(self, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(non_blocking=True, device=self.device)
        projection = self.layer['(1) projection'](vector)
        projection = self.layer['(2) projection'](projection)
        projection = self.layer['(3) projection'](projection)
        # projection # b, c, 1, 1
        projection = torch.flatten(projection, 1, -1)
        location = self.layer['location'](projection)
        return(location)

    def getCriteria(
        self,
        batch: tensordict.TensorDict,
    ) -> tensordict.TensorDict:
        batch = batch.to(self.device, non_blocking=True)
        vector = batch['vector']
        target = batch['target']
        location = self.getLocation(vector)
        distance = torch.cdist(location, location, 2)
        mask = torch.triu(torch.ones_like(distance), diagonal=1)==1
        bias = torch.pow((target[mask] - distance[mask]), 2).sum()
        # diff = (target[mask] - distance[mask]) ** 2
        stress = bias / torch.pow(target[mask], 2).sum()
        total = stress
        criteria = tensordict.TensorDict(device=self.device)
        criteria.set('stress', stress)
        # # criteria.set('divergence', divergence)
        criteria.set('total', total)
        # criteria=0
        return(criteria)

    forward = getCriteria
    pass

