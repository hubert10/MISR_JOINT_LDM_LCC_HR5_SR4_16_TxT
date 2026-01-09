import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from .spade import SPADE

"""
    CNN-SR
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.acti(out)
        return out


class Resblock(nn.Module):
    def __init__(self, n_feat, kernel_size, stride, padding):
        super(Resblock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        identity = x
        out = self.res_block(x)
        return out + identity


class Downsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(
                in_channels=in_channels * scale * scale,
                out_channels=in_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.downsample(x)


####################hybrid-spade#################


class Content_Encoder(pl.LightningModule):
    def __init__(self, out_channel):
        super(Content_Encoder, self).__init__()

        self.first_layer_sr = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        self.first_layer_ref = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_sr = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 256 256 64
            Downsample(in_channels=32, scale=2),  # 128 128 64
        )

        self.layer2_sr = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 128 128 128
            Downsample(in_channels=32, scale=2),  # 64 64 128
        )

        self.layer3_sr = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 64 64 256
            Downsample(in_channels=32, scale=2),  # 32 32 256
        )

        self.layer1_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 256 256 64
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=32, scale=2),  # 128 128 64
        )

        self.layer2_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=128,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 128 128 128
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=32, scale=2),  # 64 64 128
        )

        self.layer3_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=256,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 64 64 256
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=32, scale=2),  # 32 32 256
        )

        self.spade1 = SPADE(32, 64)
        self.spade2 = SPADE(32, 64)
        self.spade3 = SPADE(32, 64)

        self.last_linear = nn.Conv2d(32, out_channel, 1, bias=False)

    # def forward(self, sr, hr_latent, if_sample_style, style_mean, style_std):

    def forward(self, sr):
        print("-------------------Content_Encoder----------------")
        print("sr:", sr.shape)

        # (1)cnn encoder
        # b 3 256 256 -> b 8 32 32
        sr_cond = self.first_layer_sr(sr)
        sr_cond = self.layer1_sr(sr_cond)
        sr_cond = self.layer2_sr(sr_cond)
        sr_cond = self.layer3_sr(sr_cond)
        out = self.last_linear(sr_cond)
        print("out:", out.shape)

        print("-------------------Content_Encoder----------------")

        return out


class AdaIN_Encoder(pl.LightningModule):
    def __init__(self, out_channel=3):
        super(AdaIN_Encoder, self).__init__()
        self.SR_Encoder = Content_Encoder(out_channel)

    def forward(self, SR):
        print("SR:", SR.shape)

        sr_ref_latent = self.SR_Encoder(SR)

        print("sr_ref_latent:", sr_ref_latent.shape)
        return sr_ref_latent
