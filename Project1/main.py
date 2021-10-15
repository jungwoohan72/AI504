from __future__ import print_function
#%matplotlib inline
import argparse
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
import json
# from IPython.display import HTML

# from dcgan import Generator, Discriminator
from dcgan_dropout import Generator, Discriminator

from fid_score import calculate_fid_given_paths

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "dataset/celeba"
workers = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hyper Parameters
batch_size = 128
image_size = 64

nc = 3 # RGB
nz = 100 # noise dim
ngf = 64 # generator feature
ndf = 64

num_epochs = 20
lr = 0.0002
beta1 = 0.5

# Data Loading
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Generator & Discriminator Initialization
netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc,ndf).to(device)

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.

fixed_noise = torch.randn(64, nz, 1, 1, device=device) # For visualization

# For logging and resume training
continue_train = False
PATH = './wandb/run-20210907_192815-2m331w16/'

log_dict = dict()

if continue_train:
    resume_epoch = 0
    netG.load_state_dict(torch.load('./Weights/netG_epoch_$d.pt' % resume_epoch))
    netD.load_state_dict(torch.load('./Weights/netD_epoch_$d.pt' % resume_epoch))

config = {
    "batch_size":batch_size,
    "num_epochs":num_epochs,
    "lr": lr,
    "beta1":beta1
}

wandb.init(project = 'DCGAN', config = config)

json_val = json.dumps(config)
with open(''.join((wandb.run.dir, 'config.json')), 'w') as f:
    json.dump(json_val, f)

# Training
netG.train()
netD.train()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Discriminator update
        netD.zero_grad()
        img_batch = data[0].to(device) # shape: [128, 3, 64, 64]
        b_size = img_batch.size(0)
        output = netD(img_batch).view(-1)
        label = torch.full((b_size,), real_label, dtype = torch.float, device = device)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_real = output.mean().item() # correction rate

        noise = torch.randn(b_size, nz, 1, 1, device = device)
        fake = netG(noise)
        label = label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item() # correction rate
        errD = errD_real + errD_fake
        optimizerD.step()

        # Generator update
        netG.zero_grad()
        output = netD(fake).view(-1)
        label = label.fill_(real_label)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        log_dict['Loss_D'] = errD
        log_dict['Loss_G'] = errG
        log_dict['D(x)'] = D_real
        log_dict['D(G(z))_1'] = D_G_z1
        log_dict['D(G(z))_2'] = D_G_z2
        log_dict['epoch'] = epoch
        wandb.log(log_dict)

        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_real, D_G_z1, D_G_z2))

    torch.save(netG.state_dict(), './Weights/netG_epoch_%d.pt' % epoch)
    torch.save(netD.state_dict(), './Weights/netD_epoch_%d.pt' % epoch)

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
test_noise = torch.randn(1000, nz, 1, 1, device = device)
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
