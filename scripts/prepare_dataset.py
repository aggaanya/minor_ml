"""
Dataset Preparation Utilities

Functions to organize, split, and validate image datasets.
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


def organize_images(source_dir, output_dir, class_names=['real', 'ai']):
    """
    Organize images from source directory into class folders.
    
    Assumes source_dir has subdirectories or files that can be classified.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # If source_dir has class subdirectories
    if any(os.path.isdir(os.path.join(source_dir, name)) for name in class_names):
        for class_name in class_names:
            class_dir = os.path.join(source_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for fname in tqdm(files, desc=f"Organizing {class_name}"):
                    src = os.path.join(class_dir, fname)
                    dst = os.path.join(output_dir, class_name, fname)
                    shutil.copy2(src, dst)
    else:
        # Assume all images in source_dir need manual classification
        print(f"Please organize images manually into {output_dir}/real/ and {output_dir}/ai/")


def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Directory with 'real' and 'ai' subdirectories
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai')
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'ai']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Split each class
    for class_name in ['real', 'ai']:
        source_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue
        
        # Get all image files
        files = [f for f in os.listdir(source_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle
        random.shuffle(files)
        
        # Calculate splits
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        
        # Copy files
        for fname in tqdm(train_files, desc=f"Splitting {class_name} - train"):
            src = os.path.join(source_dir, fname)
            dst = os.path.join(output_dir, 'train', class_name, fname)
            shutil.copy2(src, dst)
        
        for fname in tqdm(val_files, desc=f"Splitting {class_name} - val"):
            src = os.path.join(source_dir, fname)
            dst = os.path.join(output_dir, 'val', class_name, fname)
            shutil.copy2(src, dst)
        
        for fname in tqdm(test_files, desc=f"Splitting {class_name} - test"):
            src = os.path.join(source_dir, fname)
            dst = os.path.join(output_dir, 'test', class_name, fname)
            shutil.copy2(src, dst)
        
        print(f"{class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def validate_dataset(data_dir):
    """
    Validate dataset structure and check for issues.
    
    Args:
        data_dir: Root directory with train/val/test subdirectories
    """
    print("Validating dataset...")
    
    issues = []
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            issues.append(f"Missing {split} directory")
            continue
        
        stats[split] = {}
        
        for class_name in ['real', 'ai']:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                issues.append(f"Missing {split}/{class_name} directory")
                continue
            
            # Count valid images
            files = [f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            valid_count = 0
            invalid_files = []
            
            for fname in tqdm(files, desc=f"Checking {split}/{class_name}"):
                try:
                    img_path = os.path.join(class_dir, fname)
                    img = Image.open(img_path)
                    img.verify()  # Verify it's a valid image
                    valid_count += 1
                except Exception as e:
                    invalid_files.append((fname, str(e)))
            
            stats[split][class_name] = {
                'total': len(files),
                'valid': valid_count,
                'invalid': len(invalid_files)
            }
            
            if invalid_files:
                issues.append(f"{split}/{class_name}: {len(invalid_files)} invalid files")
                for fname, error in invalid_files[:5]:  # Show first 5
                    issues.append(f"  - {fname}: {error}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        if split in stats:
            print(f"\n{split.upper()}:")
            for class_name in ['real', 'ai']:
                if class_name in stats[split]:
                    s = stats[split][class_name]
                    print(f"  {class_name}: {s['valid']} valid, {s['invalid']} invalid")
    
    # Print issues
    if issues:
        print("\n" + "="*60)
        print("ISSUES FOUND")
        print("="*60)
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No issues found!")
    
    return stats, issues


def main():
    parser = argparse.ArgumentParser(description='Dataset preparation utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize images into class folders')
    organize_parser.add_argument('--source_dir', type=str, required=True, help='Source directory')
    organize_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--data_dir', type=str, required=True, help='Data directory with real/ai folders')
    split_parser.add_argument('--output_dir', type=str, required=True, help='Output directory for split')
    split_parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    split_parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    split_parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset structure')
    validate_parser.add_argument('--data_dir', type=str, required=True, help='Data directory to validate')
    
    args = parser.parse_args()
    
    if args.command == 'organize':
        organize_images(args.source_dir, args.output_dir)
    elif args.command == 'split':
        split_dataset(args.data_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
    elif args.command == 'validate':
        validate_dataset(args.data_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
