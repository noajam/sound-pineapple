
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self):
        
        self.convLayer = nn.Conv2d()
        self.batchnorm = nn.BatchNorm2d()
        self.lrelu = nn.LeakyReLU()
        pass
        
        
    def forward(self, x):
        x = self.convLayer(x)
        x = self.batchnorm(x)
        x = self.lrelu(x)
        return x
    
"""

# NOT IN USE