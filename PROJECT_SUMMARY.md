# Project Summary: AI Image Detection for Food/Grocery Fraud Prevention

## Overview

This project implements a comprehensive machine learning system to detect AI-generated images vs real camera photos, specifically designed to prevent fraud in food/grocery product submissions.

## What Was Built

### 1. **Forensic Feature Extraction** (`src/forensic_features.py`)
   - **PRNU (Photo-Response Non-Uniformity)**: Extracts sensor noise patterns unique to real cameras
   - **CFA (Color Filter Array)**: Detects demosaicing artifacts via FFT analysis
   - **JPEG Quantization Tables**: Analyzes Q-tables to identify camera vs software origins

### 2. **Two-Stream CNN Architecture** (`src/model.py`)
   - **RGB Branch**: Pre-trained EfficientNet/ResNet backbone for image features
   - **Forensic Branch**: Smaller CNN processing extracted forensic features
   - **Fusion**: Concatenated features → FC layers → binary classification

### 3. **Training Pipeline** (`src/train.py`)
   - Transfer learning with ImageNet pre-trained weights
   - Gradual unfreezing strategy
   - Learning rate scheduling (cosine annealing)
   - Data augmentation
   - Regularization (dropout, weight decay)

### 4. **Dataset Management** (`scripts/prepare_dataset.py`)
   - Organize images into class folders
   - Split dataset into train/val/test
   - Validate dataset structure and integrity

### 5. **Evaluation & Inference**
   - **Evaluation** (`src/evaluate.py`): Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
   - **Inference** (`src/inference.py`): Single image prediction with confidence scores

### 6. **AI Image Generation** (`scripts/generate_ai_images.py`)
   - Helper script to generate AI images using Stable Diffusion
   - Supports custom prompts and batch generation

## Project Structure

```
ml_project_college/
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
├── dataset_guide.md         # Detailed dataset creation guide
├── requirements.txt         # Python dependencies
├── prompts_example.txt      # Example prompts for AI generation
│
├── src/                     # Core source code
│   ├── forensic_features.py # Forensic feature extraction
│   ├── model.py            # CNN architecture
│   ├── dataset.py          # Custom dataset class
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── inference.py        # Inference script
│
├── scripts/                 # Utility scripts
│   ├── prepare_dataset.py   # Dataset preparation
│   └── generate_ai_images.py # AI image generation
│
├── train.py                 # Training wrapper (run from root)
├── evaluate.py              # Evaluation wrapper
└── inference.py             # Inference wrapper
```

## Key Features

### Forensic Analysis
- **PRNU Noise Residuals**: Identifies unique camera sensor patterns
- **CFA Demosaicing Artifacts**: Detects periodic pixel correlations from real cameras
- **JPEG Q-Tables**: Analyzes quantization tables to distinguish camera vs software origins

### Model Architecture
- **Backbone Options**: EfficientNet-B0/B3, ResNet-50/34, MobileNet
- **Two-Stream Design**: RGB + Forensic features for robust detection
- **Transfer Learning**: Pre-trained on ImageNet, fine-tuned on your data

### Training Strategy
- **Gradual Unfreezing**: Start with classifier, gradually unfreeze backbone
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **Data Augmentation**: Rotation, color jitter, JPEG compression simulation
- **Regularization**: Dropout, batch normalization, weight decay

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Dataset
See `dataset_guide.md` for detailed instructions. Quick version:
- Collect real images (from cameras/phones)
- Generate AI images (using `scripts/generate_ai_images.py`)
- Organize into `data/train/real/` and `data/train/ai/`
- Split into train/val/test using `scripts/prepare_dataset.py`

### 3. Train Model
```bash
python train.py --data_dir data --epochs 50 --batch_size 32
```

### 4. Evaluate
```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir data
```

### 5. Predict
```bash
python inference.py --image path/to/image.jpg --model_path checkpoints/best_model.pth
```

## Expected Performance

With a good dataset (2000+ images per class):
- **Accuracy**: 85-95%+
- **ROC-AUC**: 0.90-0.98+
- **Training Time**: 2-6 hours on GPU

## Technical Highlights

1. **Robust Feature Extraction**: Multiple forensic signals (PRNU, CFA, Q-tables) complement visual features
2. **Generalization**: Designed to work across different generators (Stable Diffusion, DALL-E, etc.)
3. **Production-Ready**: Includes evaluation, inference, and dataset validation tools
4. **Flexible Architecture**: Supports multiple backbones and can disable forensic features for speed

## Limitations & Future Work

### Current Limitations
- Q-table features require JPEG format (PNG won't have Q-tables)
- Performance may degrade on heavily compressed images
- Requires diverse training data for generalization

### Future Enhancements
- Diffusion-based forensics (diffusion snap-back method)
- Synthbuster frequency-domain analysis
- Explainability (Grad-CAM visualization)
- Model distillation for deployment

## Research Basis

This implementation is based on:
- Liu et al. (PRNU noise patterns)
- Forensic demosaicing analysis (CFA artifacts)
- Kornblum et al. (JPEG Q-table forensics)
- EfficientNet-based face detection studies
- Modern transfer learning practices

## Support

For questions or issues:
1. Check `QUICKSTART.md` for common issues
2. Review `dataset_guide.md` for dataset creation
3. See `README.md` for detailed documentation

---

**Built for**: Food/Grocery fraud prevention
**Model Type**: Binary classification (Real vs AI-generated)
**Framework**: PyTorch
**Status**: Production-ready
