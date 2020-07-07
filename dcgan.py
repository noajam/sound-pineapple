import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from config.loadJSON import read_json
from models import discriminator, generator
from utils.weights import weights_init
from datasets.dataset import celeba, paintings, mnist

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from pathlib import Path

# Set configuration dictionary
cfg = read_json(Path("config/config.json"))
device = torch.device('cuda' if 
                      (torch.cuda.is_available() and cfg['cuda'] and cfg['ngpu'] > 0) else 'cpu')


if cfg['use_seed']:
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])

# Set dataset and dataloader
dataset, dataloader = None, None
if cfg['dataset'] == 'celeba':
    dataset, dataloader = celeba(cfg['batch_size'], cfg['num_workers'])
elif cfg['dataset'] == 'paintings':
    dataset, dataloader = paintings(cfg['batch_size'], cfg['num_workers'])
elif cfg['dataset'] == 'mnist':
    dataset, dataloader = mnist(cfg['batch_size'], cfg['num_workers'])
else:
    raise ValueError("Dataset specified in config/config.json is not implemented.")

netG = generator.Generator(cfg['ngpu'], cfg['nz'], cfg['ngf'], cfg['nc']).to(device)
netG.apply(weights_init)

netD = discriminator.Discriminator(cfg['ngpu'], cfg['nc'], cfg['ndf']).to(device)
netD.apply(weights_init)

# x = (9 * 5)  + (6 / 3) + (4 * 2) - (18 / 6 * 4)
# at the same time, calculate (9 * 5), (6/3), (4*2), (18/6*4)
# in sequence, calculate addition and subtraction

# define loss
# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# make an optimizer
# Adam optimizers for generator and discriminator
optimizerG = optim.Adam(netG.parameters(), cfg['lr'], betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), cfg['lr'], betas=(0.5, 0.999))

real_label = 1
fake_label = 0

gLoss = []
dLoss = []

fixed_noise = torch.randn(cfg['grid_output'], cfg['nz'], 1, 1, device=device)


# train the discriminator
# train the generator


for epoch in range(cfg['epochs']):
    
    for i, data in enumerate(dataloader):
        # loads tensor of size [batch_size, 3, 64, 64]
        
        # maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        
        data = data[0].to(device)
        batchSize = data.size(0)
        
        
        labels = torch.full((batchSize, 1, 1, 1), real_label, dtype=torch.float, device=device)
        output = netD(data)
        
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()
        
        
        noise = torch.randn(batchSize, cfg['nz'], 1, 1, device=device)
        
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
            print("\n\n---------------------------------------------------")
            print("Epoch: " + str(epoch + 1) + "/" + str(cfg['epochs']))
            print("Batch: " + str(i + 1) + "/" + str(len(dataloader)))
            print("Discriminator Error: " + "{0:.2f}".format(errD.item()))
            print("Generator Error: " + "{0:.2f}".format(errG.item()))
            print("Dx: " + "{0:.2f}".format(D_x))
            print("DGz1: " + "{0:.2f}".format(D_G_z1))
            print("DGz2: " + "{0:.2f}".format(D_G_z2))
            print("---------------------------------------------------\n")
            
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                plt.figure(figsize=(16,16))
                plt.axis("off")
                plt.title("Training Outputs: e{}, b{}".format(epoch + 1, i + 1))
                plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
                plt.show(block=False)

            print("\n50-Batch Progress:")
        else:
            progress = i % 50
            left = 49 - progress
            sys.stdout.write("\r|" + ("#" * progress) + (" " * left) + "|")
            
        if i == 0:
            torch.save(netG, 'checkpoints/generator.pth')
            torch.save(netD, 'checkpoints/discriminator.pth')
    
        
        gLoss.append(errG.item())
        dLoss.append(errD.item())
        
torch.save(netG, 'checkpoints/generator.pth')
torch.save(netD, 'checkpoints/discriminator.pth')
