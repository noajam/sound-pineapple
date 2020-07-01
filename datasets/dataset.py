from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms




def celebA():
    dataset = datasets.ImageFolder(root="data/celebA",
                                  transform=transforms.Compose([
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 , 0.5),
                                                           (0.5, 0.5, 0.5))
                                      ]))

    dataloader = DataLoader(dataset, 128, shuffle=True, num_workers=2)
    
    return dataset, dataloader


def paintings():
    
    
    return


"""
/data/celebA
    -> img_align_celeba
        -> img01.jpg
        -> img02.jpg
        -> ...
"""

# Find an image dataset of your own