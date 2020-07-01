import torch
import torch.nn as nn

from config.loadJSON import read_json

from models import discriminator, generator

from utils.weights import weights_init

from datasets.dataset import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set configuration dictionary
cfg = read_json("config/config.json")

# Set dataset and dataloader
dataset, dataloader = None, None
if cfg['dataset'] == 'celebA':
    dataset, dataloader = celebA()
elif cfg['dataset'] == 'paintings':
    dataset, dataloader == paintings()
else:
    raise ValueError("Dataset specified in config/config.json is not implemented.")

netG = generator.Generator(cfg['ngpu'], cfg['nz'], cfg['ngf'], cfg['nc'])
netG.apply(weights_init)

netD = discriminator.Discriminator(cfg['ngpu'], cfg['nc'], cfg['ndf'])
netD.apply(weights_init)

# define loss
# Binary Cross Entropy Loss
# make an optimizer
# Adam optimizers for generator and discriminator

# train the discriminator
# train the generator