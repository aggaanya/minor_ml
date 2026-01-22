"""
Inference Script for Single Image Prediction

Predict whether a single image is AI-generated or real.
"""

import torch
import argparse
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

try:
    from .model import create_model
    from .forensic_features import extract_all_forensic_features, prepare_forensic_tensor
except ImportError:
    from model import create_model
    from forensic_features import extract_all_forensic_features, prepare_forensic_tensor


def predict_image(model, image_path, device, image_size=224, use_forensic=True):
    """
    Predict if an image is AI-generated or real.
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: torch device
        image_size: Target image size
        use_forensic: Whether to use forensic features
        
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {'error': f"Failed to load image: {e}"}
    
    # Transform for RGB branch
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract forensic features
    forensic_tensor = None
    if use_forensic:
        try:
            img_np = np.array(image).astype(np.uint8)
            features = extract_all_forensic_features(img_np, image_path)
            forensic_tensor = prepare_forensic_tensor(features, (image_size, image_size))
            forensic_tensor = torch.from_numpy(forensic_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
            forensic_tensor = (forensic_tensor - forensic_tensor.mean()) / (forensic_tensor.std() + 1e-8)
        except Exception as e:
            print(f"Warning: Forensic extraction failed: {e}")
            if use_forensic:
                forensic_tensor = torch.zeros(1, 7, image_size, image_size).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor, forensic_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    
    prob_real = probs[0, 0].item()
    prob_ai = probs[0, 1].item()
    prediction = 'AI-Generated' if pred.item() == 1 else 'Real'
    
    return {
        'prediction': prediction,
        'probability_real': prob_real,
        'probability_ai': prob_ai,
        'confidence': max(prob_real, prob_ai),
        'forensic_features': {
            'prnu_std': features.get('prnu_std', 0.0) if use_forensic else None,
            'cfa_score': features.get('cfa_score', 0.0) if use_forensic else None,
            'has_custom_qtable': features.get('has_custom_table', False) if use_forensic else None
        } if use_forensic else None
    }


def main():
    parser = argparse.ArgumentParser(description='Predict if image is AI-generated or real')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                       choices=['efficientnet_b0', 'efficientnet_b3', 'resnet50', 'resnet34', 'mobilenet'],
                       help='Backbone architecture')
    parser.add_argument('--use_forensic', action='store_true', default=True,
                       help='Use forensic features branch')
    parser.add_argument('--no_forensic', dest='use_forensic', action='store_false',
                       help='Disable forensic features branch')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for prediction')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
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
    
    print(f"Model loaded. Backbone: {backbone}, Forensic: {use_forensic}\n")
    
    # Predict
    result = predict_image(
        model, args.image, device, args.image_size, use_forensic
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nProbabilities:")
    print(f"  Real: {result['probability_real']:.4f}")
    print(f"  AI:   {result['probability_ai']:.4f}")
    
    if result['forensic_features']:
        print(f"\nForensic Features:")
        print(f"  PRNU Std: {result['forensic_features']['prnu_std']:.4f}")
        print(f"  CFA Score: {result['forensic_features']['cfa_score']:.4f}")
        print(f"  Custom Q-Table: {result['forensic_features']['has_custom_qtable']}")
    
    print("="*60)


if __name__ == '__main__':
    main()
