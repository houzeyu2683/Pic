import torch
import os
import safetensors.torch
import bitsandbytes
import visualization
import tqdm
import torchvision
import tensordict

class Framework:

    def __init__(
        self, 
        model: torch.nn.Module, 
        device: str,
        history: str
    ) -> None:
        self.model = model
        self.device = device
        self.history = history
        return

    # def saveWeight(self, path: str) -> bool:
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     safetensors.torch.save_file(self.model.state_dict(), path)
    #     return(True)

    def saveCheckpoint(self, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file(self.model.state_dict(), path)
        return(True)

    def getMemory(self) -> str:
        freeness, total = torch.cuda.mem_get_info()
        occupancy = total - freeness
        value = round((occupancy / total) * 100, 2)
        memory = f"{value}%"
        return(memory)

    def fitWeight(
        self, 
        data: torch.utils.data.DataLoader,
        snapshot: int,
        total: int,
        accumulation: int,
        validation: torch.utils.data.DataLoader
    ) -> bool:
        # optimization
        rate = 1e-5
        optimization = bitsandbytes.optim.AdamW(
            self.model.parameters(), lr=rate
        )
        optimization.zero_grad()
        # dashboard
        dashboard = visualization.Dashboard(self.history)
        dashboard.openSession()
        #
        self.model.train()
        gradient = torch.amp.GradScaler()
        number = 1
        while(True):
            iteration = tqdm.tqdm(data)
            for batch in iteration:
                memory = self.getMemory()
                iteration.set_postfix({"Memory": memory})
                # scale = 0.001
                with torch.amp.autocast(self.device):
                    criteria = self.model(batch)
                    pass
                loss = torch.div(criteria['total'], accumulation)
                gradient.scale(loss).backward()
                if(number%accumulation==0):
                    # gradient.unscale_(optimization)
                    # torch.nn.utils.clip_grad_norm_(
                    #     self.model.parameters(), 
                    #     0.5
                    # )
                    gradient.step(optimization)
                    # schedule.step()
                    gradient.update()
                    optimization.zero_grad()
                    pass
                element = {
                    'Stress': criteria['stress'],
                    # 'Divergence': criteria['divergence'],
                    'Total': criteria['total'],
                }
                dashboard.insertStatistic(
                    'Loss/Data',
                    element,
                    number
                )
                # Validation
                self.model.eval()
                with torch.no_grad():
                    batch = next(iter(validation))
                    criteria = self.model(batch)
                    pass
                self.model.train()
                element = {
                    'Stress': criteria['stress'],
                    # 'Divergence': criteria['divergence'],
                    'Total': criteria['total'],
                }
                dashboard.insertStatistic(
                    'Loss/Validation',
                    element,
                    number
                )
                # Snapshot
                if((number==1) or (number)%snapshot==0):
                    path = os.path.join(
                        self.history, 
                        'weight',
                        f'{number}.pt'
                    )
                    self.saveCheckpoint(path)
                    pass
                number += 1
                termination = False if(total==-1) else (total<number)
                if(termination): break
                continue
            _ = iteration
            if(termination): break
            continue
        dashboard.closeSession()
        return(True)

    def saveWeight(self, checkpoint: list) -> bool:
        # folder = os.path.join(self.history, 'weight')
        index = checkpoint.pop(0)
        path = os.path.join(self.history, 'checkpoint', index)
        aggregate = {}
        iteration = safetensors.torch.load_file(path).items()
        for key, value in iteration:
            aggregate.update({key: value.clone()})
            continue
        _ = iteration
        #
        iteration = checkpoint
        for index in iteration:
            path = os.path.join(self.history, 'checkpoint', index)
            state = safetensors.torch.load_file(path)
            for key in aggregate: aggregate[key] += state[key]
            continue
        _ = iteration
        size = len(checkpoint) + 1
        for key in aggregate: aggregate[key] /= size
        structure = self.model.state_dict()
        assert aggregate.keys() == structure.keys()
        weight = aggregate
        path = os.path.join(self.history, 'weight.pt')
        safetensors.torch.save_file(weight, path)
        return(True)

    pass

