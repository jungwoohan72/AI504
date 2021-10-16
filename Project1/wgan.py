import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, gen_filter, num_channel):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear( input_dim, gen_filter * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(gen_filter * 8 * 4 * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(gen_filter * 8, gen_filter * 4, 5, 2, padding = 2, output_padding = 1, bias=False),
            nn.BatchNorm2d(gen_filter * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( gen_filter * 4, gen_filter * 2, 5, 2, padding = 2, output_padding = 1, bias=False),
            nn.BatchNorm2d(gen_filter * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( gen_filter * 2, gen_filter, 5, 2, padding = 2, output_padding = 1, bias=False),
            nn.BatchNorm2d(gen_filter),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(gen_filter, gen_filter // 2, 5, 2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(gen_filter),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( gen_filter // 2, num_channel, 5, 2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.init_weight()

    def forward(self, input):
        return self.main(input)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # print('here')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self, input_dim, dis_filter):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_dim, dis_filter // 2, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(dis_filter // 2, dis_filter, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(dis_filter, affine = True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(dis_filter, dis_filter * 2, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(dis_filter, affine = True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dis_filter * 2, dis_filter * 4, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(dis_filter, affine = True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dis_filter * 4, dis_filter * 8, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(dis_filter, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(dis_filter * 8, 1, 4, 1, 0, bias=False),
        )

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        return self.main(input)