"""
metrics.py — Evaluation metrics for measuring model performance.

Includes PSNR and SSIM calculations using torchmetrics. Both metrics require
data in the [0, 1] range rather than the [-1, 1] training normalization.
"""

import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor from the network's [-1, 1] training range back to [0, 1]
    for accurate metric calculation.
    """
    return (tensor + 1.0) / 2.0


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher values represent lower error between the prediction and target.
    Typical values range from 25 to 35 dB.
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    
    return peak_signal_noise_ratio(pred_01, target_01, data_range=1.0).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Structural Similarity Index Measure (SSIM).
    SSIM evaluates luminance, contrast, and structural information, providing a 
    metric that generally correlates better with human visual perception than PSNR.
    Values range from [0, 1], where 1.0 represents perfect structural similarity.
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    
    return structural_similarity_index_measure(pred_01, target_01, data_range=1.0).item()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Calculates both PSNR and SSIM for a batch of images and returns the values in a dictionary.
    """
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
    }


if __name__ == "__main__":
    identical = torch.zeros(2, 3, 64, 64)
    metrics_identical = evaluate_batch(identical, identical)
    print(f"Identical images → PSNR: {metrics_identical['psnr']:.2f} dB, SSIM: {metrics_identical['ssim']:.4f}")
    assert metrics_identical["ssim"] > 0.999, "SSIM for identical images should be ~1.0"

    pred = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    target = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    metrics_random = evaluate_batch(pred, target)
    print(f"Random images    → PSNR: {metrics_random['psnr']:.2f} dB, SSIM: {metrics_random['ssim']:.4f}")
    assert metrics_random["psnr"] < 20, "PSNR for random images should be low"

    print("\nMetrics test PASSED.")
