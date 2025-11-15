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
    
class Encoder_small(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Input 1x28x28x28
        self.enc1 = ConvBlock(in_channels, 32) # -> 32x28x28x28
        self.pool1 = nn.MaxPool3d(2)           # -> 32x14x14
        self.enc2 = ConvBlock(32, 64)          # -> 64x14x14x14
        self.pool2 = nn.MaxPool3d(2)           # -> 64x7x7
        self.bottleneck = ConvBlock(64, 128)   # -> 128x7x7x7
    
    def forward(self, x):

        s1 = self.enc1(x) # -> 32x28x28x28
        p1 = self.pool1(s1) # -> 32x14x14x14
        s2 = self.enc2(p1) # -> 64x14x14x14
        p2 = self.pool2(s2) # -> 64x7x7x7
        b = self.bottleneck(p2) # -> 128x7x7x7
        return b, s1, s2
##
###
#### ---------------------------------------------------------- Decoders for simple ----------------------------------------------------------
###
## Decoder blocks for multitask_simple
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

##
###
#### ---------------------------------------------------------- Multitask Simple Model ----------------------------------------------------------
###
## Multitask simple net
class MultiTaskNet_simple(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, latent_dim=256):
        super().__init__()
        
        # Commen encoder
        self.encoder = Encoder_small(in_channels)
        
        # Feature vector for rnn input
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.to_latent_vec = nn.Linear(128, latent_dim) # -> Bx256x

        # First decoder head for segmentation with skipped connect
        self.seg_decoder = Seg_decoder(num_classes=num_classes)
        #self.sigmoid = nn.Sigmoid()

        # Second decoder head reconstruction without skipped
        self.recon_decoder = Recon_decoder(in_channels=in_channels)
        

    def forward(self, x):
        # commen encoder
        b, s1, s2 = self.encoder(x)

        # Vectorize bottleneck output 
        pooled_vec = self.global_pool(b).view(b.size(0), -1) # -<Bx128
        latent_z = self.to_latent_vec(pooled_vec)            # ->Bx256

        # Segmentation decoder head with skips
        seg_output = self.seg_decoder(b, s1, s2)
        #seg_output = self.sigmoid(seg_output)

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

##
###
#### ---------------------------------------------------------- Multitask Big Decoders ----------------------------------------------------------
###
## Convolutional blocks for multitask_big
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
    
##
###
#### ---------------------------------------------------------- Multitask Big Model ----------------------------------------------------------
###
## Multitask big net
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

##
###
#### ---------------------------------------------------------- Attention decoders ----------------------------------------------------------
###
## for attention mechanism small model
class Seg_decoder_ag(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up_seg1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) # -> 64x14x14x14
        self.attn1 = AttentionGate3D(F_g=64, F_l=64, F_int=32)
        self.dec_seg1 = ConvBlock(128, 64) # add skip connection 64+64= 128
        
        self.up_seg2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2) # 32x28x28x28
        self.attn2 = AttentionGate3D(F_g=32, F_l=32, F_int=16)
        self.dec_seg2 = ConvBlock(64, 32) # add skip connection 32+32= 64
        
        self.out_seg = nn.Conv3d(32, num_classes, kernel_size=1) # Output segmentation
        #self.sigmoid = nn.Sigmoid()

    def forward(self, b , s1, s2):
         # segmentation decoder forward step
        us1 = self.up_seg1(b) # -> Bx64x14x14x14
        s2 = self.attn1(g=us1, x=s2)  # ag
        ds1 = self.dec_seg1(torch.cat([us1, s2], dim=1)) # Concat skip 2
        
        us2 = self.up_seg2(ds1) # -> Bx32x28x28x28
        s1 = self.attn2(g=us2, x=s1)  # ag
        ds2 = self.dec_seg2(torch.cat([us2, s1], dim=1)) # Concat skip 1
        us3 = self.out_seg(ds2) 
        #us3 = self.sigmoid(us3)
        return us3
    
class Recon_decoder_ag(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_recon1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) # ->64x14x14x14
        self.attn1_recon = AttentionGate3D(F_g=64, F_l=64, F_int=32)
        self.dec_recon1 = ConvBlock(128, 64)

        self.up_recon2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2) # -> 32x28x28x28
        self.attn2_recon = AttentionGate3D(F_g=32, F_l=32, F_int=16)
        self.dec_recon2 = ConvBlock(64, 32)

        self.out_recon = nn.Sequential(
            nn.Conv3d(32, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, b, s1, s2):
        ur1 = self.up_recon1(b)   # -> Bx64x14x14x14
        s2 = self.attn1_recon(g=ur1, x=s2)  # ag
        dr1 = self.dec_recon1(torch.cat([ur1, s2], dim=1))
        
        ur2 = self.up_recon2(dr1) # -> Bx3x28x28x28
        s1 = self.attn2_recon(g=ur2, x=s1)  # ag
        dr2 = self.dec_recon2(torch.cat([ur2, s1], dim=1))

        ur3 = self.out_recon(dr2) # -> Bx1x28x28x28

        return ur3
    
##
###
#### ---------------------------------------------------------- Attention Gate Model ----------------------------------------------------------
###
## Multitask simple net
class MultiTaskNet_ag(nn.Module):
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

        # First decoder head for segmentation with skipped connect
        self.seg_decoder = Seg_decoder_ag(num_classes=num_classes)

        # Second decoder head reconstruction without skipped
        self.recon_decoder = Recon_decoder_ag(in_channels=in_channels)

    def forward(self, x):
        # commen encoder
        s1 = self.enc1(x)       # -> Bx32x28x28x28]
        p1 = self.pool1(s1)     # -> Bx32x14x14x14
        
        s2 = self.enc2(p1)      # -> Bx64x14x14x14
        p2 = self.pool2(s2)     # -> Bx64x7x7x7x
        
        # bottleneck -> could be variational
        b = self.bottleneck(p2) # [B, 128, 7, 7, 7]

        # Segmentation decoder head with skips
        seg_output = self.seg_decoder(b, s1, s2)

        # Reconstruction decoder head without skips
        recon_output = self.recon_decoder(b, s1, s2)
        
        return seg_output, recon_output

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
## attention gate module

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Channel count of the Gating signal (from deeper layer).
            F_l (int): Channel count of the Skip connection (from encoder).
            F_int (int): Channel count of the intermediate (bottleneck) layer.
        """
        super(AttentionGate3D, self).__init__()

        # 1x1x1 conv for the gating signal (g)
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        # 1x1x1 conv for the skip connection (x)
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        # 1x1x1 conv for the combined signal to get the attention map
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g (torch.Tensor): Gating signal from the deeper layer (was upsampled).
            x (torch.Tensor): Skip connection from the encoder.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Multiply the original skip connection (x) by the attention map
        return x * psi
    

##
###
#### ---------------------------------------------------------- Fine tune model ----------------------------------------------------------
###
## Fine tune model

class AutoencoderNet(nn.Module):
    """
    Your Phase 1 model. Just an Encoder and a Recon_decoder.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder_small(in_channels)
        # Assumes Recon_decoder is defined and uses attention
        self.recon_decoder = Recon_decoder(in_channels)

    def forward(self, x):
        b, s1, s2 = self.encoder(x)
        recon_output = self.recon_decoder(b, s1, s2)
        return recon_output

class SegmentationNet(nn.Module):
    """
    Your Phase 2 model. A (pre-trained) Encoder and a Seg_decoder.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # We use the same Encoder definition as before
        self.encoder = Encoder_small(in_channels) 
        # Assumes Seg_decoder is defined and uses attention
        self.seg_decoder = Seg_decoder(num_classes)

    def forward(self, x):
        b, s1, s2 = self.encoder(x)
        seg_output = self.seg_decoder(b, s1, s2)
        return seg_output
    

##
###
#### ---------------------------------------------------------- VAE ----------------------------------------------------------
###
##

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256, NUM_CLASSES=3):
        super(VAE, self).__init__()

        self.enc = Encoder_big(in_channels)
        
        self.bottleneck = ConvBlock(128, 128) # -> 256x4x4x4

        self.z_mu = nn.Conv3d(128, latent_dim, kernel_size=1)
        self.z_logvar = nn.Conv3d(128, latent_dim, kernel_size=1)

        self.decoder_seg = Seg_decoder_big(num_classes=NUM_CLASSES)
        self.decoder_recon = Recon_decoder_big(in_channels=in_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x, s1, s2, s3 = self.enc(x)
        b = self.bottleneck(x)
        mu = self.z_mu(b)
        logvar = self.z_logvar(b)
        z = self.reparameterize(mu, logvar)
        seg_output = self.decoder_seg(z, s1, s2, s3)
        recon_output = self.decoder_recon(z, s1, s2, s3)
        return seg_output, recon_output, mu, logvar
    


