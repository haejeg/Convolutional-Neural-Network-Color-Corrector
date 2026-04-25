## Convolutional-Neural-Network-Color-Corrector

This project trains a depth-4 U-Net retouching model on paired images (original → expert edited) and runs inference on new photos.

The model predicts **global color/exposure adjustments** (and a **1‑channel local exposure map**) that are applied back onto the input image, which helps reduce color bleeding and keeps outputs stable.

### Dataset
- **Original dataset**: `https://www.kaggle.com/datasets/weipengzhang/adobe-fivek/`
- **Dataset used here**: `https://www.kaggle.com/datasets/jesucristo/mit-5k-basic`

Expected folder structure (default):

```
Data/
  Original/   # input photos
  ExpertC/    # target retouched photos (paired by filename)
Input/        # photos you want to retouch at runtime (infer.py default)
experiments/   # training runs (metrics, checkpoints, samples)
```

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py --data_dir Data --epochs 100 --batch_size 8
```

Common options:

```bash
# faster iteration
python src/train.py --epochs 50 --batch_size 8 --num_workers 4 --limit 100
```

Training outputs are written to a run folder (default: `experiments/run_YYYYMMDD_HHMMSS/`):
- `config.json` — the exact CLI args used
- `splits.json` — the exact train/val/test file lists
- `results/metrics.csv` — per-epoch metrics (includes PSNR, SSIM, and mean ΔE / CIE76)
- `results/test_metrics.json` — final test metrics for the best checkpoint
- `checkpoints/best.pth`, `checkpoints/last.pth`
- `results/epoch_XXX.jpg` — periodic visual grids `[Input | Pred | Target | Difference]`

#### Data preprocessing & augmentation (what the code actually does)
- **Train**: random crop to 384×384, random horizontal flip, input-only color jitter, input-only synthetic chromatic aberration (random green channel roll)
- **Val/Test**: center crop to 384×384 (deterministic)
- **Normalization**: all tensors are normalized to **[-1, 1]**

### Inference

Run on a single image (looks in `Input/` by default):

```bash
python infer.py photo.jpg --checkpoint checkpoints/best.pth --output results/photo.jpg
```

Process the whole input folder:

```bash
python infer.py --output results
```

Inference preserves full images by padding to a multiple of 16 (U-Net constraint) and cropping back to the original size.

### Runtime grading “preferences” (optional)

These are applied **after** the model output at runtime:

```bash
python infer.py photo.jpg --warmth 0.2 --tint 0.1 --sat 0.3 --contrast 0.1 --exposure 0.0
```

If you want warmth/tint to behave more like photo editors, enable linear mode:

```bash
python infer.py photo.jpg --warmth 0.2 --tint 0.1 --grade_linear
```

### Metrics
- **PSNR**: pixel-level fidelity (higher is better)
- **SSIM**: structural similarity (higher is better)
- **ΔE (CIE76)**: perceptual color difference in CIELAB (lower is better)

### Controlled experiments (variants)

You can run multiple controlled variants and save all results:

```bash
python src/run_variants.py --variants experiments/variants.json
```

This writes a summary CSV to `experiments/summary_*.csv` and each variant run to its own folder under `experiments/`.
