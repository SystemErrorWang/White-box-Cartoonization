import torch
import torch.nn as nn
import torch.nn.functional as F


class Whitebox(nn.Module):
    "Whitebox Cartoonizer"

    def __init__(self, channel=32):
        super(Whitebox, self).__init__()
        self.generator = Unet_Generator(channel=channel)

    def forward(self, x, r=1, eps=5e-3):
        y = self.generator(x)
        x_shape = x.shape
        dev = x.device
        N = self.box_filter(torch.ones(
            (1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r).to(dev)

        mean_x = self.box_filter(x, r) / N
        mean_y = self.box_filter(y, r) / N
        cov_xy = self.box_filter(x*y, r) / N - mean_x*mean_y
        var_x = self.box_filter(x*x, r) / N - mean_x*mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A*mean_x

        mean_A = self.box_filter(A, r) / N
        mean_b = self.box_filter(b, r) / N

        output = mean_A * x + mean_b

        return output

    def box_filter(self, x, r):
        k_size = int(2*r+1)
        b, d, _, _ = x.shape
        dev = x.device
        kernel = torch.ones((d, 1, k_size, k_size),
                            dtype=x.dtype).to(dev)/(k_size**2)
        return F.conv2d(x, kernel, bias=None, stride=1, padding=1, dilation=1, groups=d)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x + inputs


class Unet_Generator(nn.Module):
    "Whitebox Cartoonization UNet pytorch model"

    def __init__(self, channel=32):
        super(Unet_Generator, self).__init__()
        self.Conv = nn.Conv2d(3, channel, 7, 1, 3)
        self.Conv_1 = nn.Conv2d(channel, channel, 3, 2,
                                0)
        self.Conv_2 = nn.Conv2d(channel, 2*channel, 3,
                                1, 1)
        self.Conv_3 = nn.Conv2d(2*channel, 2*channel, 3,
                                2, 0)
        self.Conv_4 = nn.Conv2d(2*channel, 4*channel, 3,
                                1, 1)
        self.block_0 = ResBlock(4*channel)
        self.block_1 = ResBlock(4*channel)
        self.block_2 = ResBlock(4*channel)
        self.block_3 = ResBlock(4*channel)
        self.Conv_5 = nn.Conv2d(4*channel, 2*channel, 3,
                                1, 1)
        self.Conv_6 = nn.Conv2d(2*channel, 2*channel, 3,
                                1, 1)
        self.Conv_7 = nn.Conv2d(2*channel, channel, 3,
                                1, 1)
        self.Conv_8 = nn.Conv2d(channel, channel, 3, 1,
                                1)
        self.Conv_9 = nn.Conv2d(channel, 3, 7, 1, 3)

    def forward(self, x):
        x0 = self.Conv(x)
        x0 = F.leaky_relu(x0, 0.2)
        
        x1 = self.Conv_1(tf_same_padding(x0))
        x1 = F.leaky_relu(x1, 0.2)
        x1 = self.Conv_2(x1)
        x1 = F.leaky_relu(x1, 0.2)

        x2 = self.Conv_3(tf_same_padding(x1))
        x2 = F.leaky_relu(x2, 0.2)
        x2 = self.Conv_4(x2)
        x2 = F.leaky_relu(x2, 0.2)

        x2 = self.block_0(x2)
        x2 = self.block_1(x2)
        x2 = self.block_2(x2)
        x2 = self.block_3(x2)

        x2 = self.Conv_5(x2)
        x2 = F.leaky_relu(x2, 0.2)


        x3 = tf_upsample_bilinear(x2)
        x3 = self.Conv_6(x3+x1)
        x3 = F.leaky_relu(x3, 0.2)
        x3 = self.Conv_7(x3)
        x3 = F.leaky_relu(x3, 0.2)

        x4 = tf_upsample_bilinear(x3)
        x4 = self.Conv_8(x4+x0)
        x4 = F.leaky_relu(x4, 0.2)
        x4 = self.Conv_9(x4)

        return x4


def tf_upsample_bilinear(x):
    b, c, h, w = x.shape
    dev = x.device
    upsampled = torch.zeros(b, c, h*2, w*2).to(dev)
    upsampled[:, :, ::2, ::2] = x
    x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
    upsampled[:, :, 1::2, ::2] = (
        x_pad[:, :, :-1, :-1] + x_pad[:, :, 1:, :-1])/2
    upsampled[:, :, ::2, 1::2] = (
        x_pad[:, :, :-1, :-1] + x_pad[:, :, :-1, 1:])/2
    upsampled[:, :, 1::2, 1::2] = (
        x_pad[:, :, :-1, :-1] + x_pad[:, :, 1:, 1:])/2
    return upsampled


def tf_same_padding(x, k_size=3):
    j = k_size//2
    return F.pad(x, (j-1, j, j-1, j))