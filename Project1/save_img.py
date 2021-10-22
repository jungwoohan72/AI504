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
target_epoch = 30

model = "dcgan"

def save_image_list(dataset, real, model):
    if real:
        base_path = './%s_img_%d/real' % (model, target_epoch)
    else:
        base_path = './%s_img_%d/fake' % (model, target_epoch)

    dataset_path = []

    for i in range(len(dataset)):
        save_path =  f'{base_path}/image_{i}.png'
        dataset_path.append(save_path)
        vutils.save_image((dataset[i]*0.5)+0.5, save_path)

    return base_path

for i in range(100):

    # wgan_netG = models_64x64.Generator(nz,f_dim).to(device)
    dcgan_netG = Generator(nz,ngf,nc).to(device)

    # wgan_netG.load_state_dict(torch.load('./wgan_Weights_64/netG_epoch_%d.pth' % target_epoch))
    dcgan_netG.load_state_dict(torch.load('./dcgan_Weights_64_colab/netG_epoch_%d.pth' % target_epoch))

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
    # wgan_test_noise = utils.cuda(Variable(torch.randn(1000, nz)))
    dcgan_test_noise = torch.randn(1000, nz, 1, 1, device = device)

    # wgan_netG.eval()
    dcgan_netG.eval()

    with torch.no_grad():
        fake_dataset = dcgan_netG(dcgan_test_noise)
        # fake_dataset = wgan_netG(wgan_test_noise)


    # Path specification
    if not os.path.exists('./%s_img_%d' % (model, target_epoch)):
        os.mkdir('./%s_img_%d' % (model, target_epoch))

    if not os.path.exists('./%s_img_%d/real' % (model, target_epoch)):
        os.mkdir('./%s_img_%d/real' % (model, target_epoch))

    if not os.path.exists('./%s_img_%d/fake' % (model, target_epoch)):
        os.mkdir('./%s_img_%d/fake' % (model, target_epoch))

    # Saving

    real_image_path_list = save_image_list(real_dataset, True, model)
    fake_image_path_list = save_image_list(fake_dataset, False, model)

    # FID value

    fid_value = calculate_fid_given_paths(['./%s_img_%d/real' % (model, target_epoch), './%s_img_%d/fake' % (model, target_epoch)], 50, False, 2048)
    print(target_epoch)
    print(f'FID score: {fid_value}')

    target_epoch += 1