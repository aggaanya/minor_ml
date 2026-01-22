"""
CNN Model Architecture for AI Image Detection

Two-stream architecture:
1. RGB branch: Processes the original image
2. Forensic branch: Processes extracted forensic features (PRNU, CFA, Q-table)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ForensicImageClassifier(nn.Module):
    """
    Two-stream CNN classifier for AI vs Real image detection.
    
    Architecture:
    - RGB branch: EfficientNet/ResNet backbone for image features
    - Forensic branch: Smaller CNN for forensic features (PRNU, CFA, Q-table)
    - Fusion: Concatenated features → FC layers → binary output
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b3',
        num_classes: int = 2,
        dropout: float = 0.5,
        use_forensic_branch: bool = True,
        pretrained: bool = True
    ):
        super(ForensicImageClassifier, self).__init__()
        
        self.use_forensic_branch = use_forensic_branch
        
        # RGB branch: Pre-trained backbone
        if backbone_name.startswith('efficientnet'):
            if backbone_name == 'efficientnet_b0':
                backbone = models.efficientnet_b0(pretrained=pretrained)
                rgb_features = 1280
            elif backbone_name == 'efficientnet_b3':
                backbone = models.efficientnet_b3(pretrained=pretrained)
                rgb_features = 1536
            else:
                raise ValueError(f"Unknown EfficientNet variant: {backbone_name}")
            # Remove classifier
            self.rgb_backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif backbone_name.startswith('resnet'):
            if backbone_name == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
                rgb_features = 2048
            elif backbone_name == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
                rgb_features = 512
            else:
                raise ValueError(f"Unknown ResNet variant: {backbone_name}")
            # Remove final FC layer
            self.rgb_backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif backbone_name.startswith('mobilenet'):
            backbone = models.mobilenet_v3_large(pretrained=pretrained)
            rgb_features = 960
            self.rgb_backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Forensic branch: Process forensic features (PRNU residual + scalar features)
        # Input: 7 channels (3 PRNU + 4 scalar features)
        if use_forensic_branch:
            self.forensic_branch = nn.Sequential(
                nn.Conv2d(7, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            forensic_features = 128
        else:
            forensic_features = 0
        
        # Fusion and classification head
        total_features = rgb_features + forensic_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, rgb_image: torch.Tensor, forensic_features: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            rgb_image: RGB image tensor (B, 3, H, W)
            forensic_features: Optional forensic feature tensor (B, 7, H, W)
            
        Returns:
            Logits tensor (B, num_classes)
        """
        # RGB branch
        rgb_feat = self.rgb_backbone(rgb_image)
        if isinstance(rgb_feat, tuple):
            rgb_feat = rgb_feat[0]
        # Flatten if needed
        if rgb_feat.dim() > 2:
            rgb_feat = torch.flatten(rgb_feat, 1)
        
        # Forensic branch
        if self.use_forensic_branch and forensic_features is not None:
            forensic_feat = self.forensic_branch(forensic_features)
            # Concatenate features
            combined_feat = torch.cat([rgb_feat, forensic_feat], dim=1)
        else:
            combined_feat = rgb_feat
        
        # Classification
        logits = self.classifier(combined_feat)
        
        return logits


def create_model(
    backbone: str = 'efficientnet_b3',
    num_classes: int = 2,
    dropout: float = 0.5,
    use_forensic: bool = True,
    pretrained: bool = True
) -> ForensicImageClassifier:
    """
    Factory function to create model.
    
    Args:
        backbone: Backbone architecture name
        num_classes: Number of output classes (2 for binary)
        dropout: Dropout rate
        use_forensic: Whether to use forensic branch
        pretrained: Whether to use pretrained weights
        
    Returns:
        Model instance
    """
    model = ForensicImageClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        dropout=dropout,
        use_forensic_branch=use_forensic,
        pretrained=pretrained
    )
    return model


if __name__ == '__main__':
    # Test model
    model = create_model(backbone='efficientnet_b3', use_forensic=True)
    
    # Test forward pass
    rgb_input = torch.randn(2, 3, 224, 224)
    forensic_input = torch.randn(2, 7, 224, 224)
    
    output = model(rgb_input, forensic_input)
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
