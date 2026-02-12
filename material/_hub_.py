import torch
import os
import torchvision
import torchcodec
import functools
import PIL.Image
import tensordict
import safetensors.torch
import cv2
import application

def getCollation(queue: list) -> tensordict.TensorDict:
    bundle = {
        'latent vector': []
    }
    iteration = queue
    for item in iteration:
        path = str(item).replace('mp4', 'pt')
        vector = torch.load(path)
        bundle['latent vector'] += [vector]
        continue
    _ = iteration
    # bundle['target'] = torch.stack(bundle['target'], dim=0)
    vector = torch.cat(bundle['latent vector'], dim=0)
    vector = vector[
        torch.randperm(vector.size(0)).tolist()[:len(queue)]
    ]
    #
    # vector = torch.nn.functional.normalize(vector.flatten(1, -1), dim=1)
    target = torch.cdist(
        vector.flatten(1, -1), 
        vector.flatten(1, -1), 
        p=2.0
    )
    # target = torch.nn.functional.cosine_similarity(
    #     vector.flatten(1, -1)[:, None, :], 
    #     vector.flatten(1, -1)[None, :, :],
    #     dim=2
    # ) 
    #
    bundle['vector'] = vector
    bundle['target'] = target
    source = {
        "vector": bundle['vector'],
        "target": bundle['target']
    }
    size = len(queue)
    collation = tensordict.TensorDict(
        source = source,
        batch_size = size
    ).detach()
    return(collation)

class Unit(torch.utils.data.Dataset):

    def __init__(self, queue: list) -> None:
        self.queue = queue
        return
    
    def getLength(self) -> int:
        length = len(self.queue)
        return(length)

    def getItem(self, index: int) -> tuple:
        item = self.queue[index]
        return(item)

    __len__ = getLength
    __getitem__ = getItem
    pass

class Document:

    def __init__(self, path: str) -> None:
        self.path = path
        return

    def getQueue(self) -> list:
        folder = os.path.dirname(self.path)
        paper = open(self.path, 'r')
        queue = []
        iteration = paper.readlines()
        for item in iteration:
            path = item.replace("\n", "")
            queue += [os.path.join(folder, path)]
            continue
        _ = iteration
        paper.close()
        return(queue)

    pass

class Hub:

    def __init__(self) -> None:
        return

    def getData(self, number: int) -> torch.utils.data.DataLoader:
        name = 'data.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        if('link'):
            getLink = lambda item: str(item).replace("mp4", "pt")
            queue = list(map(getLink, queue))
            pass
        unit = Unit(queue)
        data = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=True,
            collate_fn=getCollation, #functools.partial(getCollation),
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        return(data)

    def getValidation(
        self, 
        number: int, 
        reproducibility: bool
    ) -> torch.utils.data.DataLoader:
        name = 'validation.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        if('link'):
            getLink = lambda item: str(item).replace("mp4", "pt")
            queue = list(map(getLink, queue))
            pass
        unit = Unit(queue)
        validation = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=not reproducibility,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=not reproducibility,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        return(validation)
    
    def getTest(
        self, 
        number: int,
        reproducibility: bool
    ) -> torch.utils.data.DataLoader:
        name = 'test.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        if('link'):
            getLink = lambda item: str(item).replace("mp4", "pt")
            queue = list(map(getLink, queue))
            pass        
        unit = Unit(queue)
        test = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=not reproducibility,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=not reproducibility
        )
        return(test)
    
    def getBatch(self, number: int) -> tensordict.TensorDict:
        name = 'data.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        data = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=True,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=True
        )
        batch = next(iter(data))
        return(batch)
    
    folder = 'material/storage/'
    pass

# .to("cuda", non_blocking=True)
        #
        # permutation = torch.randperm(len(compression)-1).tolist()[0:2]
        # interval = [min(permutation), max(permutation)]
        # gap = interval[1] - interval[0] # 間隔
        # if(gap > 24):
        #     delta = gap - 24
        #     interval[0] = interval[0] + delta
        #     pass
        # # interval = [min(permutation), max(permutation)]
        # #
        # # fragment = compression[interval[0]:interval[1], :, :, :]
        # reference = compression[interval[0], :, :, :]
        # target = compression[interval[1], :, :, :]
        # 
    # bundle['reference'] = torch.nn.utils.rnn.pad_sequence(
    #     bundle['reference']
    # ).permute(1, 0, 2, 3, 4)
    # bundle['target'] = torch.nn.utils.rnn.pad_sequence(
    #     bundle['target']
    # ).permute(1, 0, 2, 3, 4)




            # pace = interval[1] - interval[0]
        # bundle['reference'] += [reference]
        # bundle['target'] += [target]
        # bundle['pace'] += [pace]
        # bundle['head token'] += [0]
        # bundle['tail token'] += [1]

        # "pace": bundle['pace'],
        # 'head token': bundle['head token'],
        # 'tail token': bundle['tail token'],

    # bundle['head token'] = torch.IntTensor(bundle['head token'])
    # bundle['tail token'] = torch.IntTensor(bundle['tail token'])

