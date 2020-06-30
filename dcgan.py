import torch
import torch.nn as nn

print("Have not imported models yet.")
from models import discriminator, generator
print("Have imported models.")

gnet = generator.Generator(0, 100, 64, 3)
dnet = discriminator.Discriminator(0, 3, 64)