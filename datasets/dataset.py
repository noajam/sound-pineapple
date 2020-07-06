from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path



def celeba(batchSize, nw):

    dataset = datasets.ImageFolder(root=Path("data/celeba"),
                                  transform=transforms.Compose([
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
                                      ]))

    dataloader = DataLoader(dataset, batchSize, shuffle=True, num_workers=nw)
    
    return dataset, dataloader


def paintings(batchSize, nw):    
    dataset = datasets.ImageFolder(root=Path("data/paintings/VVG"),
                                  transform=transforms.Compose([
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
                                      ]))
    
    dataloader = DataLoader(dataset, batchSize, shuffle=True, num_workers=nw)
    
    return dataset, dataloader

def mnist(batchSize, nw):
    dataset = datasets.MNIST(root=Path("data/mnist"), download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ]))
    
    dataloader = DataLoader(dataset, batchSize, shuffle=True, num_workers=nw)
    
    return dataset, dataloader