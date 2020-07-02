import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils

from config.loadJSON import read_json

from models import discriminator, generator

from utils.weights import weights_init

from datasets.dataset import *

from tqdm import tqdm

import matplotlib.pyplot as plt

# Set configuration dictionary
cfg = read_json("config/config.json")

# Set dataset and dataloader
dataset, dataloader = None, None
if cfg['dataset'] == 'celebA':
    dataset, dataloader = celebA()
elif cfg['dataset'] == 'paintings':
    dataset, dataloader = paintings()
else:
    raise ValueError("Dataset specified in config/config.json is not implemented.")


netG = generator.Generator(cfg['ngpu'], cfg['nz'], cfg['ngf'], cfg['nc'])
netG.apply(weights_init)

netD = discriminator.Discriminator(cfg['ngpu'], cfg['nc'], cfg['ndf'])
netD.apply(weights_init)


# define loss
# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# make an optimizer
# Adam optimizers for generator and discriminator
optimizerG = optim.Adam(netG.parameters(), cfg['lr'], betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), cfg['lr'], betas=(0.5, 0.999))

real_label = 1
fake_label = 0

img_list = []
gLoss = []
dLoss = []

iters = 0

fixed_noise = torch.randn(64, cfg['nz'], 1, 1)



# train the discriminator
# train the generator


for epoch in tqdm(range(cfg['epochs'])):
    
    for i, data in enumerate(dataloader):
        # loads tensor of size [128, 3, 64, 64]
        
        # maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        
        data = data[0]
        batchSize = data.size(0)
        
        labels = torch.full((batchSize, 1, 1, 1), real_label, dtype=float)
        
        output = netD(data)
        
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()
        
        
        noise = torch.randn(batchSize, cfg['nz'], 1, 1)
        
        fake = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake.detach())
    
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # maximize log(D(G(z)))
        netG.zero_grad()
        labels.fill_(real_label)
        
        output = netD(fake)
        
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        optimizerG.step()
        
        
        if i % 50 == 0:
            print("----------------------------------------------------")
            print("Epoch: " + str(epoch + 1) + "/" + str(cfg['epochs']))
            print("Batch: " + str(i + 1) + "/" + str(len(dataloader)))
            print("Discriminator Error: " + str(errD.item()))
            print("Generator Error: " + str(errG.item()))
            print("Dx: " + str(D_x), "DGz1: " + str(D_G_z1), "DGz2: " + str(D_G_z2))
            print("----------------------------------------------------\n\n")
            torch.save(netG, 'checkpoints/generator.pth')
            torch.save(netD, 'checkpoints/discriminator.pth')
    
        
        gLoss.append(errG.item())
        dLoss.append(errD.item())
        
        if (iters % 500 == 0) or ((epoch == cfg['epochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    
    
    
    