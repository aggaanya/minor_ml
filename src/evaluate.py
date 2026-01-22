"""
Evaluation Script for AI Image Detection Model

Computes comprehensive metrics on test set.
"""

import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .model import create_model
    from .dataset import create_dataloaders
    from .dataset import AIImageDataset
except ImportError:
    from model import create_model
    from dataset import create_dataloaders
    from dataset import AIImageDataset


def evaluate_model(model, test_loader, device, use_forensic=True, save_plots=True, output_dir='results'):
    """Evaluate model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            forensic = batch['forensic'].to(device) if use_forensic else None
            
            outputs = model(images, forensic)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of AI class
            all_paths.extend(batch['path'])
    
    # Metrics
    test_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"  Real (0): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}, Support={support[0]}")
    print(f"  AI (1):   Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}, Support={support[1]}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'AI']))
    
    # Save plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # ROC curve (if binary)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        print(f"\nPlots saved to {output_dir}/")
    
    return {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'paths': all_paths
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI Image Detection Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default=None, help='Path to test directory (overrides data_dir)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                       choices=['efficientnet_b0', 'efficientnet_b3', 'resnet50', 'resnet34', 'mobilenet'],
                       help='Backbone architecture')
    parser.add_argument('--use_forensic', action='store_true', default=True,
                       help='Use forensic features branch')
    parser.add_argument('--no_forensic', dest='use_forensic', action='store_false',
                       help='Disable forensic features branch')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--no_plots', dest='save_plots', action='store_false', default=True,
                       help='Disable saving plots')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get args from checkpoint if available
    if 'args' in checkpoint:
        checkpoint_args = checkpoint['args']
        backbone = checkpoint_args.backbone if hasattr(checkpoint_args, 'backbone') else args.backbone
        use_forensic = checkpoint_args.use_forensic if hasattr(checkpoint_args, 'use_forensic') else args.use_forensic
    else:
        backbone = args.backbone
        use_forensic = args.use_forensic
    
    model = create_model(
        backbone=backbone,
        num_classes=2,
        dropout=0.5,
        use_forensic=use_forensic,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Backbone: {backbone}, Forensic: {use_forensic}")
    
    # Create test dataloader
    if args.test_dir:
        # Use custom test directory
        from torch.utils.data import DataLoader
        test_dataset = AIImageDataset(
            args.test_dir, split='', image_size=(args.image_size, args.image_size),
            augment=False, extract_forensic=use_forensic
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
    else:
        _, _, test_loader = create_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size),
            extract_forensic=use_forensic
        )
    
    # Evaluate
    results = evaluate_model(
        model, test_loader, device, use_forensic,
        save_plots=args.save_plots, output_dir=args.output_dir
    )
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
