from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models_64x64
import PIL.Image as Image
# import tensorboardX
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from fid_score import calculate_fid_given_paths

import utils

import wandb
import json
import os

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = utils.cuda(torch.rand(shape))
    z = x + alpha * (y - x)

    # gradient penalty
    z = utils.cuda(Variable(z, requires_grad=True))
    o = f(z)
    g = grad(o, z, grad_outputs=utils.cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp

""" gpu """
gpu_id = [0]
utils.cuda_devices(gpu_id)


""" param """
epochs = 100
batch_size = 64
n_critic = 5
lr = 0.0002
z_dim = 100
num_workers = 4
f_dim = 64


""" data """
crop_size = 108
re_size = 64

dataroot = "dataset/celeba/"
dataset = dsets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(re_size),
                               transforms.CenterCrop(re_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)

""" model """
D = models_64x64.DiscriminatorWGANGP(3)
G = models_64x64.Generator(z_dim)
utils.cuda([D, G])

continue_train = True

if continue_train:
    resume_epoch = 12
    G.load_state_dict(torch.load('./Weights/netG_epoch_%d.pt' % resume_epoch))
    D.load_state_dict(torch.load('./Weights/netD_epoch_%d.pt' % resume_epoch))

d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

z_sample = Variable(torch.randn(100, z_dim))
z_sample = utils.cuda(z_sample)

log_dict = dict()

config = {
    "batch_size":batch_size,
    "num_epochs":epochs,
    "lr": lr,
    "beta1": 0.5
}

wandb.init(project = 'DCGAN', config = config)

json_val = json.dumps(config)
with open(''.join((wandb.run.dir, 'config.json')), 'w') as f:
    json.dump(json_val, f)

for epoch in range(13, epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # step
        step = epoch * len(data_loader) + i + 1

        # set train
        G.train()

        # leafs
        imgs = Variable((imgs*0.5)+0.5)
        bs = imgs.size(0)
        z = Variable(torch.randn(bs, z_dim))
        imgs, z = utils.cuda([imgs, z])

        f_imgs = G(z)

        # train D
        r_logit = D(imgs)
        f_logit = D(f_imgs.detach())

        wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
        gp = gradient_penalty(imgs.data, f_imgs.data, D)
        d_loss = -wd + gp * 10.0

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        if step % n_critic == 0:
            # train G
            z = utils.cuda(Variable(torch.randn(bs, z_dim)))
            f_imgs = G(z)
            f_logit = D(f_imgs)
            g_loss = -f_logit.mean()

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            log_dict['Loss_G'] = g_loss.item()

        log_dict['Loss_D/wd'] = wd.item()
        log_dict['Loss_D/gp'] = gp.item()
        log_dict['epoch'] = epoch
        wandb.log(log_dict)

        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f' % (
            epoch, epochs, i, len(data_loader), d_loss.item()))

        if (i + 1) % 100 == 0:
            G.eval()
            f_imgs_sample = (G(z_sample).data + 1) / 2.0

            save_dir = './sample_images_while_training/celeba_wgan_gp'
            utils.mkdir(save_dir)
            torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)

    torch.save(G.state_dict(), './Weights/netG_epoch_%d.pt' % epoch)
    torch.save(D.state_dict(), './Weights/netD_epoch_%d.pt' % epoch)