# AI-Generated Image Detection for Food/Grocery Fraud Prevention

A comprehensive ML model to detect AI-generated images vs real camera photos, specifically designed to prevent fraud in food/grocery product submissions.

## Features

- **Forensic Feature Extraction**:
  - PRNU (Photo-Response Non-Uniformity) noise residuals
  - CFA (Color Filter Array) demosaicing artifacts via FFT
  - JPEG quantization table analysis
- **Two-Stream CNN Architecture**: RGB image + forensic features
- **Transfer Learning**: Pre-trained EfficientNet/ResNet with fine-tuning
- **Robust Training**: Learning rate scheduling, data augmentation, regularization

## Project Structure

```
ml_project_college/
├── requirements.txt
├── README.md
├── dataset_guide.md          # Detailed dataset creation guide
├── src/
│   ├── __init__.py
│   ├── forensic_features.py  # PRNU, CFA, Q-table extraction
│   ├── model.py              # CNN architecture
│   ├── dataset.py            # Custom dataset class
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation script
│   └── inference.py          # Single image prediction
├── scripts/
│   ├── prepare_dataset.py    # Dataset preparation utilities
│   └── download_samples.py   # Helper for sample data
└── notebooks/
    └── exploratory_analysis.ipynb  # Data exploration
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

See `dataset_guide.md` for detailed instructions on creating your dataset.

**Quick Summary:**
- **Real Images**: Collect photos from various cameras/phones (food/grocery items)
- **AI Images**: Generate using Stable Diffusion, DALL-E, Midjourney, etc.
- **Structure**:
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

### 3. Train Model

```bash
python src/train.py --data_dir data/ --epochs 50 --batch_size 32
```

### 4. Evaluate

```bash
python src/evaluate.py --model_path checkpoints/best_model.pth --test_dir data/test
```

### 5. Predict on Single Image

```bash
python src/inference.py --image path/to/image.jpg --model_path checkpoints/best_model.pth
```

## Dataset Creation Guide

**See `dataset_guide.md` for comprehensive instructions.**

### Key Points:

1. **Real Images Sources**:
   - Food photography from various cameras
   - Grocery product photos
   - Mix of lighting conditions, backgrounds
   - Different compression levels (JPEG quality 70-95)

2. **AI Images Sources**:
   - Stable Diffusion (v1.5, v2.1, SDXL)
   - DALL-E 2/3
   - Midjourney
   - Other diffusion models

3. **Data Balance**: Aim for 50/50 or 60/40 real/AI split

4. **Augmentation**: Applied automatically during training

## Model Architecture

- **Backbone**: EfficientNet-B3 or ResNet-50 (ImageNet pre-trained)
- **Forensic Branch**: Processes noise residuals, CFA features, Q-table stats
- **Fusion**: Concatenated features → FC layers → binary output
- **Output**: Probability of image being AI-generated

## Training Strategy

- Transfer learning from ImageNet
- Gradual unfreezing of layers
- Learning rate scheduling (cosine annealing)
- Data augmentation (rotation, color jitter, JPEG compression)
- Regularization (dropout, weight decay)

## Evaluation Metrics

- Accuracy, Precision, Recall
- ROC-AUC
- Confusion Matrix
- Per-class performance

## Limitations & Future Work

- Works best on JPEG images (Q-table features require JPEG)
- Performance may degrade on heavily compressed images
- Requires diverse training data for generalization
- Future: Diffusion-based forensics, Synthbuster frequency analysis

## References

Based on research from:
- Liu et al. (PRNU noise patterns)
- Forensic demosaicing analysis
- Kornblum et al. (JPEG Q-table forensics)
- EfficientNet-based face detection studies
