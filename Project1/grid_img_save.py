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
import numpy as np

import utils

import models_64x64

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
f_dim = 64
target_epoch = 40

model = "dcgan"

# Fixed Noise
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

wgan_netG = models_64x64.Generator(nz,f_dim).to(device)
dcgan_netG = Generator(nz,ngf,nc).to(device)

wgan_netG.load_state_dict(torch.load('./wgan_Weights_64/netG_epoch_%d.pt' % target_epoch))
dcgan_netG.load_state_dict(torch.load('./dcgan_Weights_64/netG_epoch_%d.pt' % target_epoch))

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
wgan_test_noise = utils.cuda(Variable(torch.randn(64, nz)))
dcgan_test_noise = torch.randn(64, nz, 1, 1, device = device)

wgan_netG.eval()
dcgan_netG.eval()

with torch.no_grad():
    fake_dataset = (dcgan_netG(dcgan_test_noise)+1) / 2.0
    vutils.save_image(fake_dataset, 'dcgan_grid.jpg', nrow=8)
    fake_dataset = (wgan_netG(wgan_test_noise).data + 1) / 2.0
    vutils.save_image(fake_dataset, 'wgan_grid.jpg', nrow=8)