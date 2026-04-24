# losses.py - Loss Function Definitions
# 
# This module defines the objective functions used to train the neural network.
# It includes perceptual loss (VGG), color difference loss (CIELAB), and adversarial loss (GAN).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
    """
    Computes perceptual similarity using pre-trained VGG16 feature maps.
    
    Instead of comparing pixel differences (which misalign easily), this compares the abstract 
    features of the generated and target images. It extracts features from the early and middle 
    layers of VGG16 to compare low-level edges/textures and mid-level structures.
    """

    # ImageNet normalization statistics required for VGG inputs
    _VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _VGG_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, device: torch.device):
        super().__init__()

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Layer 9 corresponds to relu2_2 (low-level textures)
        self.slice1 = nn.Sequential(*list(vgg.children())[:9])   
        # Layer 16 corresponds to relu3_3 (mid-level structures)
        self.slice2 = nn.Sequential(*list(vgg.children())[9:16]) 

        # Freeze the VGG parameters as we are only using it for feature extraction
        for param in self.parameters():
            param.requires_grad = False

        # Register tensors as buffers to handle device placement automatically
        self.register_buffer("vgg_mean", self._VGG_MEAN.view(1, 3, 1, 1))
        self.register_buffer("vgg_std", self._VGG_STD.view(1, 3, 1, 1))

        self.to(device)

    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from [-1, 1] input range to [0, 1] range, then apply ImageNet normalization
        x = (x + 1.0) / 2.0                       
        x = (x - self.vgg_mean) / self.vgg_std     
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_vgg = self._normalize_for_vgg(pred)
        target_vgg = self._normalize_for_vgg(target)

        pred_f1 = self.slice1(pred_vgg)
        target_f1 = self.slice1(target_vgg)

        pred_f2 = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)

        # Compute L1 distance between the extracted feature maps
        loss = F.l1_loss(pred_f1, target_f1) + F.l1_loss(pred_f2, target_f2)
        return loss


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """
    Differentiable conversion from RGB to CIELAB color space.
    
    CIELAB is perceptually uniform, meaning the Euclidean distance between two colors 
    in LAB space closely approximates human visual perception of color difference.
    """
    device = image.device
    
    image = (image + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    
    # Inverse sRGB gamma correction to linear light
    mask = (image > 0.04045).type_as(image)
    base = torch.clamp((image + 0.055) / 1.055, min=1e-5)
    image_linear = mask * torch.pow(base, 2.4) + (1.0 - mask) * image / 12.92
    
    # Linear sRGB to XYZ transform matrix
    matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=device, dtype=image.dtype)
    
    B, C, H, W = image_linear.shape
    image_flat = image_linear.view(B, 3, H * W)
    xyz = torch.bmm(matrix.unsqueeze(0).expand(B, -1, -1), image_flat)
    xyz = xyz.view(B, 3, H, W)
    
    # Normalize by D65 illuminant white point
    white_point = torch.tensor([0.95047, 1.00000, 1.08883], device=device, dtype=image.dtype).view(1, 3, 1, 1)
    xyz = xyz / white_point
    
    # Convert XYZ to LAB
    mask = (xyz > 0.008856).type_as(xyz)
    base_xyz = torch.clamp(xyz, min=1e-5)
    f_xyz = mask * torch.pow(base_xyz, 1.0/3.0) + (1.0 - mask) * (7.787 * xyz + 16.0 / 116.0)
    
    L = 116.0 * f_xyz[:, 1:2, :, :] - 16.0
    a = 500.0 * (f_xyz[:, 0:1, :, :] - f_xyz[:, 1:2, :, :])
    b = 200.0 * (f_xyz[:, 1:2, :, :] - f_xyz[:, 2:3, :, :])
    
    return torch.cat([L, a, b], dim=1)


def cielab_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the L1 distance between predictions and targets in CIELAB color space."""
    pred_lab = rgb_to_lab(pred)
    target_lab = rgb_to_lab(target)
    return F.l1_loss(pred_lab, target_lab)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    perceptual_loss_fn: PerceptualLoss,
    l1_weight: float = 0.5,
    cielab_weight: float = 0.5,
    perceptual_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Computes a weighted sum of spatial L1, perceptual CIELAB, and VGG feature losses.
    Returns the total scalar loss along with a dictionary of the individual components.
    """
    l1 = F.l1_loss(pred, target)
    cielab = cielab_loss(pred, target)
    perceptual = perceptual_loss_fn(pred, target)

    total = l1_weight * l1 + cielab_weight * cielab + perceptual_weight * perceptual

    return total, {
        "l1": l1.item(),
        "cielab": cielab.item(),
        "perceptual": perceptual.item(),
        "total": total.item()
    }


class GANLoss(nn.Module):
    """
    Calculates the adversarial loss for GAN training.
    
    Defaults to Least Squares GAN (LSGAN) using MSELoss, which typically exhibits
    greater training stability than the standard BCE formulation.
    """
    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


if __name__ == "__main__":
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    perceptual_fn = PerceptualLoss(device=device)

    pred = torch.randn(2, 3, 384, 384, device=device)
    target = torch.randn(2, 3, 384, 384, device=device)

    loss, components = combined_loss(pred, target, perceptual_fn)

    print(f"L1 loss:         {components['l1']:.4f}")
    print(f"CIELAB loss:     {components['cielab']:.4f}")
    print(f"Perceptual loss: {components['perceptual']:.4f}")
    print(f"Total loss:      {components['total']:.4f}")
    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert not torch.isnan(loss), "Loss must not be NaN"

    print("\nLoss test PASSED.")
