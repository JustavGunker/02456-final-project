import torch
import torch.nn as nn


INPUT_SHAPE = (128, 128, 128)
NUM_CLASSES = 3 
LATENT_DIM = 256 
BATCH_SIZE = 4

## Convolutional blocks for multitask_simple
class ConvBlock(nn.Module):
    """
    A 3D Convolutional block: Conv3D -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.3)
        )

    def forward(self, x):
        return self.conv(x)

class Seg_decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up_seg1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) # -> 64x14x14x14
        self.dec_seg1 = ConvBlock(128, 64) # add skip connection 64+64= 128
        
        self.up_seg2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2) # 32x28x28x28
        self.dec_seg2 = ConvBlock(64, 32) # add skip connection 32+32= 64
        
        self.out_seg = nn.Conv3d(32, num_classes, kernel_size=1) # Output segmentation
        
    def forward(self, b , s1, s2):
         # segmentation decoder forward step
        us1 = self.up_seg1(b) # -> Bx64x14x14x14
        ds1 = self.dec_seg1(torch.cat([us1, s2], dim=1)) # Concat skip 2
        
        us2 = self.up_seg2(ds1) # -> Bx32x28x28x28
        ds2 = self.dec_seg2(torch.cat([us2, s1], dim=1)) # Concat skip 1
        us3 = self.out_seg(ds2) 
        return us3
    
class Recon_decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_recon1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) # ->64x14x14x14
        self.dec_recon1 = ConvBlock(128, 64)

        self.up_recon2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2) # -> 32x28x28x28
        self.dec_recon2 = ConvBlock(64, 32)

        self.out_recon = nn.Sequential(
            nn.Conv3d(32, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, b, s1, s2):
        ur1 = self.up_recon1(b)   # -> Bx64x14x14x14
        dr1 = self.dec_recon1(torch.cat([ur1, s2], dim=1))
        
        ur2 = self.up_recon2(dr1) # -> Bx3x28x28x28
        dr2 = self.dec_recon2(torch.cat([ur2, s1], dim=1))

        ur3 = self.out_recon(dr2) # -> Bx1x28x28x28

        return ur3
    

## Multitask simple net
class MultiTaskNet_simple(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, latent_dim=256):
        super().__init__()
        
        # Commen encoder
        # 1x28x28x28
        self.enc1 = ConvBlock(in_channels, 32) # -> 32x28x28x28
        self.pool1 = nn.MaxPool3d(2) # -> 32x14x14x14
        
        self.enc2 = ConvBlock(32, 64) # -> 64x14x14x14
        self.pool2 = nn.MaxPool3d(2) # -> 64x7x7x7
        
        # Bottleneck 
        self.bottleneck = ConvBlock(64, 128) # -> 128x7x7x7
        
        # Feature vector for rnn input
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.to_latent_vec = nn.Linear(128, latent_dim) # -> Bx256x

        # First decoder head for segmentation with skipped connect
        self.seg_decoder = Seg_decoder(num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()

        # Second decoder head reconstruction without skipped
        self.recon_decoder = Recon_decoder(in_channels=in_channels)
        

    def forward(self, x):
        # commen encoder
        s1 = self.enc1(x)       # -> Bx32x28x28x28]
        p1 = self.pool1(s1)     # -> Bx32x14x14x14
        
        s2 = self.enc2(p1)      # -> Bx64x14x14x14
        p2 = self.pool2(s2)     # -> Bx64x7x7x7x
        
        # bottleneck -> could be variational
        b = self.bottleneck(p2) # [B, 128, 7, 7, 7]

        # Vectorize bottleneck output 
        pooled_vec = self.global_pool(b).view(b.size(0), -1) # -<Bx128
        latent_z = self.to_latent_vec(pooled_vec)            # ->Bx256

        # Segmentation decoder head with skips
        seg_output = self.seg_decoder(b, s1, s2)
        seg_output = self.sigmoid(seg_output)

        # Reconstruction decoder head without skips
        recon_output = self.recon_decoder(b, s1, s2)
        
        return seg_output, recon_output, latent_z


# Temporal model -> could add more dimension 
class TemporalTracker(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True 
        )

        # predict t+1
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, z_sequence):
        # z_sequence shape -> Batch x Time_Steps x latent_dim
        lstm_out, _ = self.lstm(z_sequence)
        
        # only care about t+1
        last_step_out = lstm_out[:, -1, :] # -1 last time step +1 
        
        # fc layer for prediction
        prediction = self.fc(last_step_out)
        return prediction
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Encoder_big(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Input: 1x32x32x32
        self.enc1 = ConvBlock(in_channels, 32) # -> 32x32x32
        self.pool1 = nn.MaxPool3d(2)           # -> 32x16x16
        
        self.enc2 = ConvBlock(32, 64)          # -> 64x16x16
        self.pool2 = nn.MaxPool3d(2)           # -> 64x8x8

        self.enc3 = ConvBlock(64, 128)         # -> 128x8x8
        self.pool3 = nn.MaxPool3d(2)           # -> 128x4x4
        
    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3) # p3 is 128x4x4
        return p3, s1, s2, s3

class Seg_decoder_big(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # b (bottleneck) is 256x4x4
        self.up_seg1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2) # -> 128x8x8
        self.dec_seg1 = ConvBlock(256, 128) # 128+128 = 256
        
        self.up_seg2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) # -> 64x16x16
        self.dec_seg2 = ConvBlock(128, 64) # 64+64 = 128

        self.up_seg3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2) # -> 32x32x32
        self.dec_seg3 = ConvBlock(64, 32) # 32+32 = 64
        
        self.out_seg = nn.Conv3d(32, num_classes, kernel_size=1)
        
    def forward(self, b , s1, s2, s3):
        us1 = self.up_seg1(b) # 8x8x8
        ds1 = self.dec_seg1(torch.cat([us1, s3], dim=1)) # cat [8x8x8] w [8x8x8]
        
        us2 = self.up_seg2(ds1) # 16x16x16
        ds2 = self.dec_seg2(torch.cat([us2, s2], dim=1)) # cat [16x16x16] w[16x16x16]

        us3 = self.up_seg3(ds2) # 32x32x32
        ds3 = self.dec_seg3(torch.cat([us3, s1], dim=1)) # cat [32x32x32] w [32x32x32]
        us4 = self.out_seg(ds3) 
        return us4
    
class Recon_decoder_big(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # b (bottleneck) is 256x4x4
        self.up_recon1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2) # -> 128x8x8
        self.dec_recon1 = ConvBlock(128+128, 128)

        self.up_recon2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) # -> 64x16x16
        self.dec_recon2 = ConvBlock(64+64, 64)

        self.up_recon3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2) # -> 32x32x32
        self.dec_recon3 = ConvBlock(32+32, 32)

        self.out_recon = nn.Sequential(
            nn.Conv3d(32, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, b, s1, s2, s3):
        ur1 = self.up_recon1(b)
        dr1 = self.dec_recon1(torch.cat([ur1, s3], dim=1))
        ur2 = self.up_recon2(dr1)
        dr2 = self.dec_recon2(torch.cat([ur2, s2], dim=1))
        ur3 = self.up_recon3(dr2)
        dr3 = self.dec_recon3(torch.cat([ur3, s1], dim=1))
        ur4 = self.out_recon(dr3)
        return ur4
    

class MultiTaskNet_big(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, latent_dim=512):
        super().__init__()
        
        # Commen encoder
        self.encoder = Encoder_big(in_channels)

        # Bottleneck 
        self.bottleneck = ConvBlock(128, 256) # -> 256x8x8x8
        
        # Feature vector for rnn input
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.to_latent_vec = nn.Linear(256, latent_dim) # -> Bx512

        # First decoder head for segmentation with skipped connect
        self.seg_decoder = Seg_decoder_big(num_classes=num_classes)

        # Second decoder head reconstruction with skipped connect
        self.recon_decoder = Recon_decoder_big(in_channels=in_channels)
        

    def forward(self, x):
        # commen encoder
        x, s1, s2, s3 = self.encoder(x)
        
        # bottleneck -> could be variational
        b = self.bottleneck(x) # [B, 256, 7, 7, 7]

        # Vectorize bottleneck output 
        pooled_vec = self.global_pool(b).view(b.size(0), -1) # -<Bx256
        latent_z = self.to_latent_vec(pooled_vec)            # ->Bx256

        # Segmentation decoder head with skips
        seg_output = self.seg_decoder(b, s1, s2, s3)

        # Reconstruction decoder head with skipped connect
        recon_output = self.recon_decoder(b, s1, s2, s3)
        
        return seg_output, recon_output, latent_z


# Temporal model -> could add more dimension 
class TemporalTracker(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True 
        )

        # predict t+1
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, z_sequence):
        # z_sequence shape -> Batch x Time_Steps x latent_dim
        lstm_out, _ = self.lstm(z_sequence)
        
        # only care about t+1
        last_step_out = lstm_out[:, -1, :] # -1 last time step +1 
        
        # fc layer for prediction
        prediction = self.fc(last_step_out)
        return prediction
