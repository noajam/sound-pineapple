from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms




def celebA():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
        ])
    
    dataset = datasets.ImageFolder(root="data/celebA",
                                  transform=transform)

    dataloader = DataLoader(dataset, 128, shuffle=True, num_workers=2)
    
    return dataset, dataloader


def paintings():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
        ])
    
    dataset = datasets.ImageFolder(root="data/paintings/VVG",
                                  transform=transform)
    
    dataloader = DataLoader(dataset, 32, shuffle=True, num_workers=2)
    
    return dataset, dataloader

def MNIST():
    dataset = datasets.MNIST(train=True)
    
def fashionMNIST():
    pass


"""
/data/celebA
    -> img_align_celeba
        -> img01.jpg
        -> img02.jpg
        -> ...
"""

# Find an image dataset of your own