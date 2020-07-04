from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms





def celeba(batchSize):

    dataset = datasets.ImageFolder(root="data/celebA",
                                  transform=transforms.Compose([
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
                                      ]))

    dataloader = DataLoader(dataset, batchSize, shuffle=True, num_workers=2)
    
    return dataset, dataloader


def paintings():    
    dataset = datasets.ImageFolder(root="data/paintings/VVG",
                                  transform=transforms.Compose([
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
                                      ]))
    
    dataloader = DataLoader(dataset, 32, shuffle=True, num_workers=2)
    
    return dataset, dataloader

def mnist(batchSize):
    dataset = datasets.MNIST(root="data/mnist", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ]))
    
    dataloader = DataLoader(dataset, batchSize, shuffle=True, num_workers=2)
    
    return dataset, dataloader