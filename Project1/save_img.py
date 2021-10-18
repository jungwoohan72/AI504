import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import models_64x64

import utils

from fid_score import calculate_fid_given_paths
from dcgan import Generator

dataroot = "dataset/celeba"
workers = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
batch_size = 128
image_size = 64

nc = 3 # RGB
nz = 100 # noise dim
ngf = 64 # generator feature
ndf = 64
f_dim = 128

target_epoch = 37

netG = G = models_64x64.Generator(nz,f_dim).to(device)
netG.load_state_dict(torch.load('./Weights/netG_epoch_%d.pt' % target_epoch))

# 1000 real images
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000,
                                         shuffle=True, num_workers=workers)

for i, data in enumerate(dataloader):
    real_dataset = data[0]
    break

# 1000 fake images
test_noise = utils.cuda(Variable(torch.randn(1000, nz)))
netG.eval()
with torch.no_grad():
    fake_dataset = netG(test_noise)

# Path specification
if not os.path.exists('/img'):
    os.mkdir('./img')

if not os.path.exists('/img/real'):
    os.mkdir('./img/real')

if not os.path.exists('/img/fake'):
    os.mkdir('./img/fake')

# Saving
base_path = ['./img/real', './img/fake']
dataset = [real_dataset, fake_dataset]

for i in range(2):
    for j in range(1000):
        save_path = f'{base_path[i]}/image_{j}.png'
        vutils.save_image((dataset[i][j]*0.5) + 0.5, save_path)

# FID value

fid_value = calculate_fid_given_paths(['./img/real', './img/fake'], 50, False, 2048)
print(f'FID score: {fid_value}')