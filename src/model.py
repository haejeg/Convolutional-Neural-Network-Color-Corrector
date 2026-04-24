"""
model.py — U-Net architecture for image-to-image photo retouching.

Architecture overview:
  The U-Net has an encoder path that progressively downsamples the image to extract
  high-level features, and a decoder path that upsamples it back to the original resolution.
  
  Skip connections are used to concatenate the high-resolution feature maps from the encoder
  directly to the corresponding layers in the decoder, preserving fine spatial details.

  Residual learning:
    Instead of outputting the final image directly, the network predicts a correction
    adjustment that is applied to the original input image. This improves training stability
    since the initial output starts close to an identity mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A foundational building block consisting of two consecutive Convolutional layers.
    Each convolution is followed by Batch Normalization to stabilize training and
    a ReLU activation function to introduce non-linearity.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Encoder step: Halves the spatial dimensions using MaxPool2d, followed by
    feature extraction using the DoubleConv block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: Doubles the spatial dimensions using bilinear upsampling,
    concatenates the corresponding skip connection from the encoder, and applies
    the DoubleConv block to merge the features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Bilinear upsampling avoids checkerboard artifacts typical with ConvTranspose2d
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad x if its spatial dimensions don't match the skip tensor due to odd sizing
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])

        # Concatenate along the channel dimension
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    A depth-4 U-Net model designed for image-to-image translation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        b = base_channels

        # Encoder pathway
        self.enc1 = DoubleConv(in_channels, b)        
        self.enc2 = Down(b, b * 2)                    
        self.enc3 = Down(b * 2, b * 4)                
        self.enc4 = Down(b * 4, b * 8)                

        # Bottleneck layer
        self.bottleneck = Down(b * 8, b * 8)          

        # Decoder pathway
        self.dec4 = Up(b * 8 + b * 8, b * 4)         
        self.dec3 = Up(b * 4 + b * 4, b * 2)         
        self.dec2 = Up(b * 2 + b * 2, b)             
        self.dec1 = Up(b + b, b // 2)                 

        # Final convolution layer outputs 7 channels:
        # 6 channels for global RGB Gain and Gamma adjustments.
        # 1 channel for local spatial exposure adjustments.
        self.final_conv = nn.Conv2d(b // 2, 7, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_input = x  # Store input for the final residual application

        # Encoder forward pass with skip connections saved
        s1 = self.enc1(x)          
        s2 = self.enc2(s1)         
        s3 = self.enc3(s2)         
        s4 = self.enc4(s3)         

        # Bottleneck forward pass
        neck = self.bottleneck(s4)

        # Decoder forward pass using skip connections
        x = self.dec4(neck, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Output parameter map
        x_out = self.final_conv(x)  

        # Compute global tone adjustments (Gain and Gamma) by averaging spatially
        global_params = torch.mean(x_out[:, 0:6, :, :], dim=(2, 3), keepdim=True)
        
        gain_logits = global_params[:, 0:3, :, :]
        gamma_logits = global_params[:, 3:6, :, :]

        # Constrain parameter ranges
        gain = torch.sigmoid(gain_logits) * 1.5 + 0.2
        gamma = torch.exp(gamma_logits / 2.0)

        # Extract the local exposure map
        exposure_logits = x_out[:, 6:7, :, :]
        exposure_map = torch.sigmoid(exposure_logits) + 0.5 

        # Transform raw input to [0, 1] range for color math
        x_01 = torch.clamp((raw_input + 1.0) / 2.0, min=1e-6, max=1.0)
        
        # Apply color adjustments: Gamma correction, Gain, and Local Exposure mapping
        x_01 = (x_01 ** gamma) * gain * exposure_map
            
        # Re-normalize back to [-1, 1] range
        output = torch.clamp((x_01 * 2.0) - 1.0, -1.0, 1.0)
        return output


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.
    
    Instead of outputting a single real/fake scalar for the entire image, it outputs 
    a matrix of predictions evaluating whether each N x N local image patch is real or fake.
    This encourages the generator to produce high-frequency details.
    """
    def __init__(self, in_channels: int = 6, base_channels: int = 64):
        # in_channels is 6 because the input and target images are concatenated
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    model.eval()

    dummy_input = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}") 
    print(f"Output shape: {output.shape}")       
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]") 
    print(f"Trainable parameters: {count_parameters(model):,}")        

    assert output.shape == dummy_input.shape, "Output shape must match input shape!"
    print("\nModel test PASSED.")
