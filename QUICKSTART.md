# Quick Start Guide

This guide will help you get started quickly with the AI Image Detection model.

## Step 1: Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Note**: For generating AI images, you'll also need:
```bash
pip install diffusers transformers accelerate
```

## Step 2: Prepare Your Dataset

### Option A: You already have images organized

If you have images in folders like:
```
my_images/
├── real/
│   ├── img1.jpg
│   └── img2.jpg
└── ai/
    ├── img1.jpg
    └── img2.jpg
```

Split them into train/val/test:
```bash
python scripts/prepare_dataset.py split --data_dir my_images --output_dir data --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

### Option B: Generate AI images

1. Generate AI images using Stable Diffusion:
```bash
python scripts/generate_ai_images.py --output_dir data/ai_raw --num_images 10
```

2. Collect real images (from your camera, phone, etc.) and put them in `data/real_raw/`

3. Organize and split:
```bash
# First organize
mkdir -p data/organized
cp -r data/real_raw/* data/organized/real/
cp -r data/ai_raw/* data/organized/ai/

# Then split
python scripts/prepare_dataset.py split --data_dir data/organized --output_dir data --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

### Option C: Use existing dataset

If you have a dataset already split:
```
data/
├── train/
│   ├── real/
│   └── ai/
├── val/
│   ├── real/
│   └── ai/
└── test/
    ├── real/
    └── ai/
```

Validate it:
```bash
python scripts/prepare_dataset.py validate --data_dir data
```

## Step 3: Train the Model

```bash
python train.py --data_dir data --epochs 50 --batch_size 32 --backbone efficientnet_b3
```

**Key arguments:**
- `--data_dir`: Path to your data directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32, reduce if out of memory)
- `--backbone`: Model architecture (`efficientnet_b3`, `efficientnet_b0`, `resnet50`, etc.)
- `--lr`: Learning rate (default: 1e-4)
- `--use_forensic`: Use forensic features (default: True)
- `--no_forensic`: Disable forensic features (faster, less accurate)

**Monitor training:**
```bash
tensorboard --logdir checkpoints/logs
```

## Step 4: Evaluate

After training, evaluate on test set:
```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir data
```

This will show:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Step 5: Predict on Single Image

```bash
python inference.py --image path/to/image.jpg --model_path checkpoints/best_model.pth
```

## Common Issues & Solutions

### Out of Memory (OOM)

- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use smaller model: `--backbone efficientnet_b0`
- Reduce image size: `--image_size 224` (default is 224)

### Slow Training

- Use GPU if available (CUDA)
- Reduce number of workers: `--num_workers 2`
- Disable forensic features: `--no_forensic` (less accurate but faster)

### Poor Accuracy

- Collect more training data (aim for 1000+ images per class)
- Ensure balanced dataset (similar number of real and AI images)
- Train for more epochs
- Try different backbones
- Enable forensic features if disabled

### Dataset Issues

- Validate your dataset: `python scripts/prepare_dataset.py validate --data_dir data`
- Check image formats (JPEG preferred for Q-table features)
- Ensure images are at least 224x224 pixels

## Expected Results

With a good dataset (2000+ images per class):
- **Accuracy**: 85-95%+
- **ROC-AUC**: 0.90-0.98+
- **Training time**: 2-6 hours on GPU (depending on dataset size)

## Next Steps

- Experiment with different backbones
- Try different learning rates
- Add more data augmentation
- Fine-tune hyperparameters
- Test on different image types

Good luck! 🚀
