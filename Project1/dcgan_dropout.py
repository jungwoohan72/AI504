import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, gen_filter, num_channel):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_dim, gen_filter * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_filter * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(gen_filter * 8, gen_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filter * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( gen_filter * 4, gen_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filter * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( gen_filter * 2, gen_filter, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filter),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( gen_filter, num_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, input_dim, dis_filter):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_dim, dis_filter, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(dis_filter, dis_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(dis_filter * 2, dis_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dis_filter * 4, dis_filter * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(dis_filter * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)