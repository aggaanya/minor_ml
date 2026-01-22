"""
Training Script for AI Image Detection Model

Implements:
- Transfer learning with gradual unfreezing
- Learning rate scheduling (cosine annealing)
- Data augmentation
- Regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

try:
    from .model import create_model
    from .dataset import create_dataloaders
except ImportError:
    from model import create_model
    from dataset import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, use_forensic=True):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        forensic = batch['forensic'].to(device) if use_forensic else None
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, forensic)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, use_forensic=True):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
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
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return epoch_loss, epoch_acc, precision, recall, f1, auc


def main():
    parser = argparse.ArgumentParser(description='Train AI Image Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3', 
                       choices=['efficientnet_b0', 'efficientnet_b3', 'resnet50', 'resnet34', 'mobilenet'],
                       help='Backbone architecture')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--use_forensic', action='store_true', default=True, 
                       help='Use forensic features branch')
    parser.add_argument('--no_forensic', dest='use_forensic', action='store_false',
                       help='Disable forensic features branch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'plateau'], help='LR scheduler type')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        extract_forensic=args.use_forensic
    )
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    model = create_model(
        backbone=args.backbone,
        num_classes=2,
        dropout=args.dropout,
        use_forensic=args.use_forensic,
        pretrained=True
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Gradual unfreezing strategy
    # Start with only classifier trainable
    for param in model.rgb_backbone.parameters():
        param.requires_grad = False
    
    # Optimizer (only trainable parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)
    else:  # plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Gradual unfreezing: unfreeze more layers every few epochs
        if epoch == args.epochs // 3:
            print("Unfreezing more layers...")
            for param in list(model.rgb_backbone.children())[-3:]:
                for p in param.parameters():
                    p.requires_grad = True
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=1e-4
            )
            if args.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-6)
        
        if epoch == args.epochs * 2 // 3:
            print("Unfreezing all layers...")
            for param in model.rgb_backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.01,  # Even lower LR
                weight_decay=1e-4
            )
            if args.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-6)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.use_forensic
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device, args.use_forensic
        )
        
        # Update learning rate
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Metrics/Precision', val_prec, epoch)
        writer.add_scalar('Metrics/Recall', val_rec, epoch)
        writer.add_scalar('Metrics/F1', val_f1, epoch)
        writer.add_scalar('Metrics/AUC', val_auc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'args': args
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val_acc: {best_val_acc:.4f}")
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
