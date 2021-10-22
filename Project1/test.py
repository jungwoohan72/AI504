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


def save_image_list(dataset, real, target_idx):
    base_path = './test/img_%d' % (target_idx)

    dataset_path = []

    for i in range(len(dataset)):
        save_path =  f'{base_path}/image_{i}.png'
        dataset_path.append(save_path)
        vutils.save_image((dataset[i]*0.5)+0.5, save_path)

    return base_path

dataroot = "dataset/celeba"
image_size = 64
workers = 4
target_idx = 0

for i in range(20):
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

    if not os.path.exists('./test/img_%d' % (target_idx)):
        os.mkdir('./test/img_%d' % (target_idx))

    real_image_path_list = save_image_list(real_dataset, True, target_idx)

    fid_value = calculate_fid_given_paths(['./test/img_%d' % (target_idx), './test/img_42/fake'], 50, False, 2048)
    print(target_idx)
    print(f'FID score: {fid_value}')

    target_idx += 1