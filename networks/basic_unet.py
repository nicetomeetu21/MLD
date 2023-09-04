""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mynet_parts.normlizations import get_norm_layer


class UNet(nn.Module):
    def __init__(self, n_in=1, n_out=1, first_channels=64, n_dps=4, use_bilinear=False, use_pool=False,
                 norm_type='instance'):
        super(UNet, self).__init__()

        norm_layer = get_norm_layer(norm_type)

        self.encoder = UnetEncoder2d(n_in, first_channels, n_dps, use_pool, norm_layer)
        first_channels = first_channels * pow(2, n_dps)
        self.decoder = UnetDecoder2d(n_out, first_channels, n_dps, use_bilinear, norm_layer)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


class UnetEncoder2d(nn.Module):
    def __init__(self, in_channels, first_channels, n_dps, use_pool, norm_layer):
        super(UnetEncoder2d, self).__init__()
        self.inc = InConv(in_channels, first_channels, norm_layer)
        self.down_blocks = nn.ModuleList()
        in_channels = first_channels
        out_channels = in_channels * 2
        for i in range(n_dps):
            self.down_blocks.append(Down(in_channels, out_channels, use_pool, norm_layer))
            in_channels = out_channels
            out_channels = in_channels * 2

    def forward(self, x):
        x = self.inc(x)
        out_features = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            out_features.append(x)
        return out_features


class UnetDecoder2d(nn.Module):
    def __init__(self, n_classes, first_channels, n_dps, use_bilinear, norm_layer, out_feat = False):
        super(UnetDecoder2d, self).__init__()
        self.out_feat = out_feat
        self.up_blocks = nn.ModuleList()
        T_channels = first_channels
        out_channels = T_channels // 2
        in_channels = T_channels + out_channels

        for i in range(n_dps):
            self.up_blocks.append(Up(T_channels, in_channels, out_channels, use_bilinear, norm_layer))
            T_channels = out_channels
            out_channels = T_channels // 2
            in_channels = T_channels + out_channels
        # one more divide in out_channels
        self.outc = nn.Conv2d(out_channels*2, n_classes, kernel_size=1)

    def forward(self, features):

        pos_feat = len(features) - 1
        x = features[pos_feat]
        if self.out_feat:
            ret_feat = []
            ret_feat.append(x)
        for up_block in self.up_blocks:
            pos_feat -= 1
            x = up_block(x, features[pos_feat])
            if self.out_feat:
                ret_feat.append(x)
        x = self.outc(x)
        if self.out_feat:
            return x, ret_feat
        else:
            return x

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.double_conv = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool, norm_layer):
        super().__init__()
        if use_pool:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                nn.ReLU(inplace=True),
                norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
                nn.ReLU(inplace=True)
            )
        else:
            self.down_conv = nn.Sequential(
                norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
                nn.ReLU(inplace=True),
                norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, T_channels, in_channels, out_channels, bilinear, norm_layer):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(T_channels, T_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
