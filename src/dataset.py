"""
Custom Dataset Class for AI Image Detection

Handles loading images with forensic feature extraction.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable
import cv2

try:
    from .forensic_features import extract_all_forensic_features, prepare_forensic_tensor
except ImportError:
    from forensic_features import extract_all_forensic_features, prepare_forensic_tensor


class AIImageDataset(Dataset):
    """
    Dataset for AI vs Real image classification.
    
    Loads images and extracts forensic features on-the-fly.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        extract_forensic: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing train/val/test subdirectories
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            augment: Whether to apply data augmentation (only for training)
            extract_forensic: Whether to extract forensic features
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.extract_forensic = extract_forensic
        
        # Load image paths
        real_dir = os.path.join(data_dir, split, 'real')
        ai_dir = os.path.join(data_dir, split, 'ai')
        
        self.image_paths = []
        self.labels = []
        
        # Real images (label 0)
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(real_dir, fname))
                    self.labels.append(0)
        
        # AI images (label 1)
        if os.path.exists(ai_dir):
            for fname in os.listdir(ai_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(ai_dir, fname))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.image_paths)} images for {split} split "
              f"({sum(1 for l in self.labels if l == 0)} real, "
              f"{sum(1 for l in self.labels if l == 1)} AI)")
        
        # Data augmentation (only for training)
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/test: only resize and normalize
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
            - 'image': RGB image tensor (3, H, W)
            - 'forensic': Forensic feature tensor (7, H, W) or None
            - 'label': Class label (0=real, 1=AI)
            - 'path': Image path
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', self.image_size, (0, 0, 0))
        
        # Apply transforms (augmentation + normalization)
        image_tensor = self.transform(image)
        
        # Extract forensic features
        forensic_tensor = None
        if self.extract_forensic:
            try:
                # Convert PIL to numpy for forensic extraction
                img_np = np.array(image).astype(np.uint8)
                
                # Extract features
                features = extract_all_forensic_features(img_np, image_path)
                
                # Convert to tensor format
                forensic_tensor = prepare_forensic_tensor(features, self.image_size)
                
                # Convert to torch tensor and normalize
                forensic_tensor = torch.from_numpy(forensic_tensor).permute(2, 0, 1)  # (C, H, W)
                
                # Normalize forensic features
                forensic_tensor = (forensic_tensor - forensic_tensor.mean()) / (forensic_tensor.std() + 1e-8)
                
            except Exception as e:
                # If forensic extraction fails, create zero tensor
                print(f"Warning: Forensic extraction failed for {image_path}: {e}")
                forensic_tensor = torch.zeros(7, self.image_size[0], self.image_size[1])
        
        return {
            'image': image_tensor,
            'forensic': forensic_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'path': image_path
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    extract_forensic: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory with train/val/test subdirectories
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        extract_forensic: Whether to extract forensic features
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = AIImageDataset(
        data_dir, split='train', image_size=image_size,
        augment=True, extract_forensic=extract_forensic
    )
    val_dataset = AIImageDataset(
        data_dir, split='val', image_size=image_size,
        augment=False, extract_forensic=extract_forensic
    )
    test_dataset = AIImageDataset(
        data_dir, split='test', image_size=image_size,
        augment=False, extract_forensic=extract_forensic
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        dataset = AIImageDataset(data_dir, split='train', extract_forensic=True)
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Forensic shape: {sample['forensic'].shape}")
            print(f"Label: {sample['label']}")
        else:
            print("No images found in dataset")
    else:
        print("Usage: python dataset.py <data_dir>")
