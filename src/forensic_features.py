"""
Forensic Feature Extraction for AI Image Detection

This module implements:
1. PRNU (Photo-Response Non-Uniformity) noise residual extraction
2. CFA (Color Filter Array) demosaicing artifact detection via FFT
3. JPEG quantization table extraction and analysis
"""

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage, signal
from scipy.fft import fft2, fftshift
import warnings
from typing import Tuple, Optional, Dict
import io

try:
    import jpeglib
    JPEGLIB_AVAILABLE = True
except ImportError:
    JPEGLIB_AVAILABLE = False
    warnings.warn("jpeglib not available. Q-table extraction will be limited.")


def extract_prnu_residual(image: np.ndarray, method: str = 'wiener') -> np.ndarray:
    """
    Extract PRNU (Photo-Response Non-Uniformity) noise residual from image.
    
    Real camera images contain unique high-frequency sensor noise patterns,
    while AI generators produce smoother output. This extracts the noise residual.
    
    Args:
        image: Input image as numpy array (H, W, 3) in range [0, 255]
        method: 'wiener' or 'highpass' for noise extraction
        
    Returns:
        Noise residual as numpy array (H, W, 3)
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    if method == 'wiener':
        # Wiener filter approach (Liu et al.)
        # Apply a denoising filter and subtract from original
        denoised = cv2.fastNlMeansDenoisingColored(
            (img_float * 255).astype(np.uint8),
            None, 10, 10, 7, 21
        ).astype(np.float32) / 255.0
        
        residual = img_float - denoised
        
    elif method == 'highpass':
        # High-pass filter approach
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        residual = np.zeros_like(img_float)
        for c in range(3):
            residual[:, :, c] = signal.convolve2d(
                img_float[:, :, c], kernel, mode='same', boundary='symm'
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize residual
    residual = (residual - residual.mean()) / (residual.std() + 1e-8)
    
    return residual


def extract_cfa_features(image: np.ndarray, block_size: int = 64) -> Dict[str, float]:
    """
    Extract CFA (Color Filter Array) demosaicing artifacts via FFT analysis.
    
    Real cameras use Bayer CFA and demosaicing, creating periodic pixel correlations.
    AI images skip this pipeline, so they lack true demosaicing patterns.
    
    Args:
        image: Input image as numpy array (H, W, 3) in range [0, 255]
        block_size: Size of blocks for FFT analysis (default 64)
        
    Returns:
        Dictionary with CFA-related features:
        - cfa_score: Aggregated periodic pattern strength
        - green_residual_energy: Energy in green channel residuals
        - periodic_strength: Strength of periodic components
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Focus on green channel (richest in Bayer CFA)
    green = img_float[:, :, 1]
    
    # Compute prediction residual (simple high-pass)
    kernel = np.array([[0, -1, 0],
                      [-1,  4, -1],
                      [0, -1, 0]]) / 4.0
    green_residual = signal.convolve2d(
        green, kernel, mode='same', boundary='symm'
    )
    
    # Divide into blocks and compute FFT
    h, w = green_residual.shape
    scores = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = green_residual[i:i+block_size, j:j+block_size]
            
            # Compute 2D FFT
            fft_block = fft2(block)
            fft_magnitude = np.abs(fftshift(fft_block))
            
            # Look for periodic patterns (high energy in mid-frequencies)
            # Real demosaicing creates periodic spikes
            center = block_size // 2
            # Exclude DC component and very high frequencies
            mask = np.ones((block_size, block_size), dtype=bool)
            mask[center-2:center+3, center-2:center+3] = False  # Exclude DC
            mask[:center-8, :] = False  # Exclude very low freq
            mask[center+8:, :] = False  # Exclude very high freq
            
            # Compute energy in periodic frequency range
            periodic_energy = np.sum(fft_magnitude[mask])
            scores.append(periodic_energy)
    
    if len(scores) == 0:
        # Fallback for small images
        fft_magnitude = np.abs(fftshift(fft2(green_residual)))
        center = fft_magnitude.shape[0] // 2
        mask = np.ones_like(fft_magnitude, dtype=bool)
        mask[center-2:center+3, center-2:center+3] = False
        cfa_score = np.sum(fft_magnitude[mask])
    else:
        cfa_score = np.mean(scores)
    
    # Additional features
    green_residual_energy = np.sum(green_residual ** 2)
    periodic_strength = np.std(scores) if len(scores) > 1 else 0.0
    
    return {
        'cfa_score': float(cfa_score),
        'green_residual_energy': float(green_residual_energy),
        'periodic_strength': float(periodic_strength)
    }


def extract_jpeg_qtable(image_path: str) -> Optional[Dict[str, any]]:
    """
    Extract JPEG quantization table from image.
    
    Camera firmware and editing software use distinctive Q-tables.
    AI tools often produce JPEGs with default or uniform Q-tables.
    
    Args:
        image_path: Path to JPEG image
        
    Returns:
        Dictionary with Q-table features, or None if not available:
        - qtable_luma: Luminance Q-table (8x8 array)
        - qtable_chroma: Chrominance Q-table (8x8 array)
        - qtable_sum: Sum of Q-table values
        - qtable_high_freq: Sum of high-frequency coefficients
        - has_custom_table: Whether Q-table differs from standard
    """
    try:
        if JPEGLIB_AVAILABLE:
            # Use jpeglib for accurate extraction
            jpeg = jpeglib.read_dct(image_path)
            
            qtable_luma = jpeg.qt[0]  # Luminance table
            qtable_chroma = jpeg.qt[1] if len(jpeg.qt) > 1 else jpeg.qt[0]  # Chrominance
            
            # Compute features
            qtable_sum = np.sum(qtable_luma) + np.sum(qtable_chroma)
            
            # High-frequency coefficients (bottom-right of 8x8 block)
            high_freq_luma = np.sum(qtable_luma[4:, 4:])
            high_freq_chroma = np.sum(qtable_chroma[4:, 4:])
            qtable_high_freq = high_freq_luma + high_freq_chroma
            
            # Check if custom (compare to standard tables)
            # Standard JPEG tables are typically uniform or have specific patterns
            std_luma = np.std(qtable_luma)
            has_custom = std_luma > 5.0  # Threshold for custom tables
            
            return {
                'qtable_luma': qtable_luma.tolist(),
                'qtable_chroma': qtable_chroma.tolist(),
                'qtable_sum': float(qtable_sum),
                'qtable_high_freq': float(qtable_high_freq),
                'has_custom_table': bool(has_custom),
                'qtable_std': float(std_luma)
            }
        else:
            # Fallback: Try to extract from PIL/OpenCV
            # This is less accurate but works for basic analysis
            img = Image.open(image_path)
            
            # If image was just opened, Q-table info might be in EXIF
            # For now, return None and rely on other features
            # In production, you'd parse JPEG markers directly
            return None
            
    except Exception as e:
        # Q-table extraction failed (PNG, corrupted JPEG, etc.)
        return None


def extract_all_forensic_features(image: np.ndarray, image_path: Optional[str] = None) -> Dict[str, any]:
    """
    Extract all forensic features from an image.
    
    Args:
        image: Image as numpy array (H, W, 3) in range [0, 255]
        image_path: Optional path to image file (for Q-table extraction)
        
    Returns:
        Dictionary with all extracted features
    """
    features = {}
    
    # PRNU residual
    prnu_residual = extract_prnu_residual(image, method='wiener')
    features['prnu_residual'] = prnu_residual
    features['prnu_std'] = float(np.std(prnu_residual))
    features['prnu_mean'] = float(np.mean(prnu_residual))
    
    # CFA features
    cfa_features = extract_cfa_features(image)
    features.update(cfa_features)
    
    # JPEG Q-table (if path provided and JPEG)
    if image_path and image_path.lower().endswith(('.jpg', '.jpeg')):
        qtable_features = extract_jpeg_qtable(image_path)
        if qtable_features:
            features.update(qtable_features)
        else:
            # Default values when Q-table unavailable
            features['qtable_sum'] = 0.0
            features['qtable_high_freq'] = 0.0
            features['has_custom_table'] = False
            features['qtable_std'] = 0.0
    else:
        features['qtable_sum'] = 0.0
        features['qtable_high_freq'] = 0.0
        features['has_custom_table'] = False
        features['qtable_std'] = 0.0
    
    return features


def prepare_forensic_tensor(features: Dict[str, any], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Prepare forensic features as a tensor for model input.
    
    Converts extracted features into a format suitable for concatenation
    with RGB image tensor.
    
    Args:
        features: Dictionary from extract_all_forensic_features
        target_size: Target image size (H, W)
        
    Returns:
        Forensic feature tensor (H, W, C) where C includes:
        - PRNU residual (3 channels)
        - Scalar features tiled across spatial dimensions
    """
    h, w = target_size
    
    # PRNU residual (resize if needed)
    prnu = features['prnu_residual']
    if prnu.shape[:2] != (h, w):
        prnu = cv2.resize(prnu, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize PRNU to [0, 1] range
    prnu = (prnu - prnu.min()) / (prnu.max() - prnu.min() + 1e-8)
    
    # Create additional feature maps from scalar features
    # Normalize scalar features to [0, 1] range (rough estimates)
    cfa_score = np.clip(features['cfa_score'] / 1000.0, 0, 1)
    qtable_sum = np.clip(features['qtable_sum'] / 10000.0, 0, 1)
    qtable_high_freq = np.clip(features['qtable_high_freq'] / 5000.0, 0, 1)
    has_custom = 1.0 if features['has_custom_table'] else 0.0
    
    # Tile scalar features across spatial dimensions
    feature_maps = np.zeros((h, w, 4), dtype=np.float32)
    feature_maps[:, :, 0] = cfa_score
    feature_maps[:, :, 1] = qtable_sum
    feature_maps[:, :, 2] = qtable_high_freq
    feature_maps[:, :, 3] = has_custom
    
    # Concatenate: PRNU (3) + scalar features (4) = 7 channels
    forensic_tensor = np.concatenate([prnu, feature_maps], axis=2)
    
    return forensic_tensor.astype(np.float32)
