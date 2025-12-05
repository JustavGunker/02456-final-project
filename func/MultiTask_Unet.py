import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def double_conv(in_channels, out_channels):
    
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(Encoder, self).__init__()
        self.conv_down1 = double_conv(in_channels, 16)
        self.conv_down2 = double_conv(16, 32)
        self.conv_down3 = double_conv(32, 64)
        self.conv_down4 = double_conv(64, 128)
        self.conv_down5 = double_conv(128, 256)
        self.conv_down6 = double_conv(256, 512)
        self.conv_down7 = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv_down1(x)
        x2 = self.maxpool(x1)
        x3 = self.conv_down2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv_down3(x4)
        x6 = self.maxpool(x5)
        x7 = self.conv_down4(x6)
        x8 = self.maxpool(x7)
        x9 = self.conv_down5(x8)
        x10 = self.maxpool(x9)
        x11 = self.conv_down6(x10)
        x12 = self.maxpool(x11)
        bott = self.conv_down7(x12)

        # return feature maps for skip connections
        return bott, [x1, x3, x5, x7, x9, x11]


class ReconstructionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = double_conv(128, 64)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec5 = double_conv(64, 32)
        self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec6 = double_conv(32, 16)

        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, bott, skips):
        x1, x2, x3, x4, x5, x6 = skips

        u1 = self.up1(bott)
        u1 = self.dec1(torch.cat([u1, x6], dim=1))

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, x5], dim=1))

        u3 = self.up3(u2)
        u3 = self.dec3(torch.cat([u3, x4], dim=1))

        u4 = self.up4(u3)
        u4 = self.dec4(torch.cat([u4, x3], dim=1))

        u5 = self.up5(u4)
        u5 = self.dec5(torch.cat([u5, x2], dim=1))

        u6 = self.up6(u5)
        u6 = self.dec6(torch.cat([u6, x1], dim=1))

        return self.out(u6)
    
class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = double_conv(128, 64)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec5 = double_conv(64, 32)
        self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec6 = double_conv(32, 16)

        self.out = nn.Conv2d(16, num_classes, 1)

    def forward(self, bott, skips):
        x1, x2, x3, x4, x5, x6 = skips

        u1 = self.up1(bott)
        u1 = self.dec1(torch.cat([u1, x6], dim=1))

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, x5], dim=1))

        u3 = self.up3(u2)
        u3 = self.dec3(torch.cat([u3, x4], dim=1))

        u4 = self.up4(u3)
        u4 = self.dec4(torch.cat([u4, x3], dim=1))

        u5 = self.up5(u4)
        u5 = self.dec5(torch.cat([u5, x2], dim=1))

        u6 = self.up6(u5)
        u6 = self.dec6(torch.cat([u6, x1], dim=1))

        return self.out(u6)
    
class SemiSupervisedUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = Encoder()
        self.recon_decoder = ReconstructionDecoder()
        self.seg_decoder = SegmentationDecoder(num_classes)

    def forward(self, x, do_segmentation=True):
        bott, skips = self.encoder(x)
        recon = self.recon_decoder(bott, skips)
        seg = self.seg_decoder(bott, skips) if do_segmentation else None
        return recon, seg
