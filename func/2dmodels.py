import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv_unpadded(in_channels, out_channels):
    """
    Two consecutive 3x3 unpadded convolutions (padding=0).
    Total size reduction per block is 4 pixels (2 from first conv + 2 from second conv).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class EncoderBlock(nn.Module):
    """
    U-Net Encoder step: Conv-Conv block followed by Max Pooling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = double_conv_unpadded(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        conv_output = self.conv(x)
        pool_output = self.pool(conv_output)
        return conv_output, pool_output

class DecoderBlock(nn.Module):
    """
    U-Net Decoder step: Up-convolution, Cropping, Concatenation, and Refinement.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )
        self.conv = double_conv_unpadded(out_channels + skip_channels, out_channels)

    def forward(self, x_bottom, x_skip):
        
        x_up = self.up(x_bottom) 
        
        diff_h = x_skip.size(2) - x_up.size(2)
        diff_w = x_skip.size(3) - x_up.size(3)
        
        if diff_h < 0 or diff_w < 0:
             raise RuntimeError(f"Cropping error: x_up ({x_up.shape}) is larger than x_skip ({x_skip.shape}).")

        # Corrected cropping logic for odd/even mismatches
        start_h = diff_h // 2
        end_h = x_skip.size(2) - (diff_h - start_h)
        start_w = diff_w // 2
        end_w = x_skip.size(3) - (diff_w - start_w)

        x_skip_cropped = x_skip[:, :, start_h:end_h, start_w:end_w]
        
        x_combined = torch.cat([x_up, x_skip_cropped], dim=1)
        
        return self.conv(x_combined)

class Encoder(nn.Module):
    """
    Block-based U-Net Encoder: 5 levels of downsampling.
    The bottleneck is the final convolution block *within* the encoder.
    """
    def __init__(self):
        super().__init__()
        self.block1 = EncoderBlock(1, 64)        # 1 -> 64 channels
        self.block2 = EncoderBlock(64, 128)      # 64 -> 128 channels
        self.block3 = EncoderBlock(128, 256)     # 128 -> 256 channels
        self.block4 = EncoderBlock(256, 512)     # 256 -> 512 channels
        
        # This is the U-Net bottleneck
        self.bottleneck = double_conv_unpadded(512, 1024) 

    def forward(self, x):
        x1, p1 = self.block1(x)
        x2, p2 = self.block2(p1)
        x3, p3 = self.block3(p2)
        x4, p4 = self.block4(p3)
        x5 = self.bottleneck(p4) # x5 is the bottleneck tensor
        
        return x1, x2, x3, x4, x5

class Decoder(nn.Module):
    def __init__(self, num_classes=1): 
        super().__init__()
        # bottle be 1024 channels
        self.upconv4 = DecoderBlock(in_channels=1024, skip_channels=512, out_channels=512) 
        self.upconv3 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256) 
        self.upconv2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128) 
        self.upconv1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64) 
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1) 

    def forward(self, x5_bottleneck, x4_skip, x3_skip, x2_skip, x1_skip):
        d4 = self.upconv4(x5_bottleneck, x4_skip)
        d3 = self.upconv3(d4, x3_skip)
        d2 = self.upconv2(d3, x2_skip)
        d1 = self.upconv1(d2, x1_skip)
        
        return self.out_conv(d1)

class Unet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder
        output = self.decoder(x5, x4, x3, x2, x1)
        
        return output
