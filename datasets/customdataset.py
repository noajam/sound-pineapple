import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas as pd

from skimage import io



class Random_Dataset(Dataset):
    def __init__(self, rootcsv, rootimgs, transform=None):
        self.rootcsv = rootcsv
        self.rootimgs = rootimgs
        self.df = pd.read_csv(rootcsv)
        self.transform = transform
    
    def __len__(self):
        # how many samples of data in dataset
        return len(self.df)
        
    def __getitem__(self, index):
        # allow you to use Random_Dataset[index]
        if torch.is_tensor(index):
            index = index.tolist()
            
        pokemonName = self.df.iloc[index, 31]
        img = None
        file = None
        fileName = []
        imgList = []
        if isinstance(pokemonName, list):
            for i, name in enumerate(pokemonName):
                fileName.append(self.rootimgs + pokemonName + ".png")     
                imgList.append(io.imread(fileName[i]))
        else:
            file = self.rootimgs + pokemonName + ".png"
            img = io.imread(file)
            
        sample = None
        if img and file:
            sample = {'image': img, 'name': pokemonName}
        else:
            sample = {'image': imgList, 'name': pokemonName}
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample