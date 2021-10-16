import torch
from torch.autograd import Variable


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, nc, image_size):
    # Follows the implementation at: https://github.com/bioinf-jku/TTUR/blob/master/WGAN_GP/gan_64x64_FID.py#L576-L590

    # alpha for interpolation
    alpha = torch.rand(batch_size, 1).to(real_data)
    alpha = alpha.expand(batch_size, real_data.nelement() // batch_size).contiguous()
    alpha = alpha.view(batch_size, nc, image_size, image_size)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(real_data),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # record slopes
    slopes = gradients.norm(2, dim=1)

    gradient_penalty = ((slopes - 1.) ** 2).mean() * 10

    return gradient_penalty