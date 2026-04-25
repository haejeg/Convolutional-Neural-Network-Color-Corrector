"""
infer.py — Inference script for running the trained model on new images.

Loads a saved model checkpoint and applies the retouching network to the provided images. 
Handles required padding since the U-Net architecture requires spatial dimensions to be 
divisible by 16.

HOW TO USE:
  python infer.py photo.jpg
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from src.model import UNet
from src.utils import get_device, tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Run photo retouching inference")
    parser.add_argument("image_name", type=str, nargs="?", default=None,
                        help="Input image name (e.g., 'hi.jpg') in the 'Input' folder. Leave blank to process the whole folder.")
    parser.add_argument("--input", type=str, default=None,
                        help="Alternative way to provide input image name or path")
    parser.add_argument("--output", type=str, default="results",
                        help="Output image path or directory (defaults to 'results')")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint (default: checkpoints/best.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: 'cpu', 'cuda', or 'mps'")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    """Loads the model and checkpoint for inference."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python src/train.py"
        )

    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set evaluation mode to disable operations like dropout and batch normalization updates
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
    return model


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    """
    Pads tensor spatial dimensions to the nearest target multiple.
    This ensures compatibility with the U-Net architecture downsampling/upsampling factors.
    """
    _, _, h, w = tensor.shape
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, (pad_h, pad_w)


def preprocess_image(image_path: str, device: torch.device) -> tuple:
    """Loads image data and transforms it into the format expected by the model."""
    img = Image.open(image_path).convert("RGB")
    original_size = (img.height, img.width)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = normalize(to_tensor(img)).unsqueeze(0) 

    tensor, padding = pad_to_multiple(tensor, multiple=16)
    tensor = tensor.to(device)

    return tensor, original_size, padding


def postprocess_tensor(tensor: torch.Tensor, original_size: tuple) -> Image.Image:
    """Converts the model output tensor back into a PIL Image and crops padding."""
    tensor = tensor.squeeze(0) 

    h_orig, w_orig = original_size
    tensor = tensor[:, :h_orig, :w_orig]

    return tensor_to_pil(tensor)


@torch.no_grad() 
def retouch_image(model: UNet, image_path: str, output_path: str, device: torch.device):
    """Performs inference on a single image and saves the result."""
    print(f"  Processing: {Path(image_path).name}")

    tensor, original_size, padding = preprocess_image(image_path, device)
    pred = model(tensor)
    result_img = postprocess_tensor(pred, original_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"  Saved:      {output_path}")


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (from --device flag)")
    else:
        device = get_device()

    model = load_model(args.checkpoint, device)

    input_given = args.image_name or args.input
    if input_given:
        input_path = Path(input_given)
        if not input_path.exists() and (Path("Input") / input_given).exists():
            input_path = Path("Input") / input_given
        elif not input_path.exists() and not input_path.is_absolute():
            input_path = Path("Input") / input_given
    else:
        input_path = Path("Input")

    output_path = Path(args.output)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if input_path.is_dir():
        image_files = [p for p in sorted(input_path.iterdir())
                       if p.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in: {input_path}")
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {len(image_files)} images from {input_path}...\n")

        for img_path in image_files:
            out_file = output_path / img_path.name
            retouch_image(model, str(img_path), str(out_file), device)

        print(f"\nDone. {len(image_files)} images saved to {output_path}/")

    elif input_path.is_file():
        if input_path.suffix.lower() not in image_extensions:
            print(f"Unsupported file type: {input_path.suffix}")
            print(f"Supported: {', '.join(image_extensions)}")
            sys.exit(1)

        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / input_path.name

        print(f"\nRetouching {input_path}...\n")
        retouch_image(model, str(input_path), str(output_path), device)
        print("\nDone.")
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
