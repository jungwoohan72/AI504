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
epochs = 50
batch_size = 64
n_critic = 5
lr = 0.0002
z_dim = 100
num_workers = 4


""" data """
crop_size = 108
re_size = 64
# offset_height = (218 - crop_size) // 2
# offset_width = (178 - crop_size) // 2
# crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

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

d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

""" load checkpoint """
# ckpt_dir = './checkpoints/celeba_wgan_gp'
# utils.mkdir(ckpt_dir)
# try:
#     ckpt = utils.load_checkpoint(ckpt_dir)
#     start_epoch = ckpt['epoch']
#     D.load_state_dict(ckpt['D'])
#     G.load_state_dict(ckpt['G'])
#     d_optimizer.load_state_dict(ckpt['d_optimizer'])
#     g_optimizer.load_state_dict(ckpt['g_optimizer'])
# except:
#     print(' [*] No checkpoint!')
#     start_epoch = 0


""" run """
# writer = tensorboardX.SummaryWriter('./summaries/celeba_wgan_gp')

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

for epoch in range(0, epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # step
        step = epoch * len(data_loader) + i + 1

        # set train
        G.train()

        # leafs
        imgs = Variable(imgs)
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

        # writer.add_scalar('D/wd', wd.data.cpu().numpy(), global_step=step)
        # writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

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

            # writer.add_scalars('G',
            #                    {"g_loss": g_loss.data.cpu().numpy()},
            #                    global_step=step)

        log_dict['Loss_D'] = d_loss
        log_dict['Loss_G'] = g_loss
        log_dict['epoch'] = epoch
        wandb.log(log_dict)

        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (
            epoch, epochs, i, len(data_loader), d_loss.item(), g_loss.item()))

        if (i + 1) % 100 == 0:
            G.eval()
            f_imgs_sample = (G(z_sample).data + 1) / 2.0

            save_dir = './sample_images_while_training/celeba_wgan_gp'
            utils.mkdir(save_dir)
            torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)

    torch.save(G.state_dict(), './Weights/netG_epoch_%d.pt' % epoch)
    torch.save(D.state_dict(), './Weights/netD_epoch_%d.pt' % epoch)

# 1000 real images
dataset = dsets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(re_size),
                               transforms.CenterCrop(re_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000,
                                         shuffle=True, num_workers=num_workers)

for i, data in enumerate(dataloader):
    real_dataset = data[0]
    break

# 1000 fake images
test_noise = Variable(torch.randn(1000, z_dim))
test_noise = utils.cuda(test_noise)

G.eval()
with torch.no_grad():
    fake_dataset = G(test_noise)

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
