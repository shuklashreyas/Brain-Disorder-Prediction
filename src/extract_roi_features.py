"""
ROI Feature Extraction for Bayesian Model
Extracts region-of-interest features from MRI images for Bayesian logistic regression.
"""

import numpy as np
import cv2
from pathlib import Path
import json
from PIL import Image
import os

def extract_roi_features(image_path, roi_regions=None):
    """
    Extract ROI features from a single MRI image.
    
    Args:
        image_path: Path to MRI image
        roi_regions: List of (x, y, width, height) tuples for ROI regions.
                    If None, uses predefined brain regions.
    
    Returns:
        Dictionary of extracted features
    """
    # Load and preprocess image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Normalize image
    img = img.astype(np.float32) / 255.0
    
    # Define default ROI regions if not provided
    # These represent key brain regions: hippocampus, ventricles, cortex
    h, w = img.shape
    if roi_regions is None:
        roi_regions = [
            (w//4, h//4, w//4, h//4),      # Left hippocampus region
            (3*w//4, h//4, w//4, h//4),    # Right hippocampus region
            (w//4, 3*h//4, w//2, h//4),    # Ventricle region
            (w//4, h//2, w//2, h//4),      # Cortex region
        ]
    
    features = {}
    
    # Global features
    features['global_mean'] = np.mean(img)
    features['global_std'] = np.std(img)
    features['global_skew'] = _calculate_skewness(img)
    features['global_kurtosis'] = _calculate_kurtosis(img)
    
    # ROI-specific features
    for i, (x, y, width, height) in enumerate(roi_regions):
        roi = img[y:y+height, x:x+width]
        if roi.size == 0:
            continue
            
        prefix = f'roi_{i}'
        features[f'{prefix}_mean'] = np.mean(roi)
        features[f'{prefix}_std'] = np.std(roi)
        features[f'{prefix}_min'] = np.min(roi)
        features[f'{prefix}_max'] = np.max(roi)
        features[f'{prefix}_median'] = np.median(roi)
        
        # Texture features (using local binary patterns approximation)
        features[f'{prefix}_texture'] = _calculate_texture(roi)
        
        # Histogram features
        hist = np.histogram(roi, bins=10)[0]
        features[f'{prefix}_hist_entropy'] = _calculate_entropy(hist)
    
    # Additional global texture features
    features['global_texture'] = _calculate_texture(img)
    features['global_entropy'] = _calculate_entropy(np.histogram(img, bins=20)[0])
    
    # Convert numpy types to native Python types for JSON serialization
    return {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
            for k, v in features.items()}

def _calculate_skewness(data):
    """Calculate skewness of data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def _calculate_kurtosis(data):
    """Calculate kurtosis of data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def _calculate_texture(roi):
    """Calculate texture measure using variance of local differences."""
    if roi.size < 4:
        return 0
    # Calculate local variance as texture measure
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        return np.var(roi)
    filtered = cv2.filter2D(roi, -1, kernel)
    return np.var(filtered)

def _calculate_entropy(hist):
    """Calculate entropy from histogram."""
    hist = hist + 1e-10  # Avoid log(0)
    prob = hist / np.sum(hist)
    return -np.sum(prob * np.log2(prob))

def extract_features_from_directory(data_dir, output_path='data/roi_features.json'):
    """
    Extract ROI features from all images in directory structure.
    
    Expected structure:
    data_dir/
        NonDemented/
            *.jpg
        VeryMildDemented/
            *.jpg
        MildDemented/
            *.jpg
        ModerateDemented/
            *.jpg
    """
    data_dir = Path(data_dir)
    features_list = []
    
    # Support multiple naming conventions
    labels_map = {
        'NonDemented': 0,
        'VeryMildDemented': 1,
        'MildDemented': 2,
        'ModerateDemented': 3,
        'No Impairment': 0,
        'Very Mild Impairment': 1,
        'Mild Impairment': 2,
        'Moderate Impairment': 3
    }
    
    # Binary classification: NonDemented/No Impairment (0) vs Demented (1, 2, 3 -> 1)
    binary_labels_map = {
        'NonDemented': 0,
        'VeryMildDemented': 1,
        'MildDemented': 1,
        'ModerateDemented': 1,
        'No Impairment': 0,
        'Very Mild Impairment': 1,
        'Mild Impairment': 1,
        'Moderate Impairment': 1
    }
    
    # Check if we're in a train/test subdirectory structure
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    directories_to_process = []
    if train_dir.exists():
        directories_to_process.append(train_dir)
    if test_dir.exists():
        directories_to_process.append(test_dir)
    if not directories_to_process:
        # Try direct structure
        directories_to_process = [data_dir]
    
    for data_path in directories_to_process:
        for label_name, label in labels_map.items():
            label_dir = data_path / label_name
            if not label_dir.exists():
                continue
        
            image_files = list(label_dir.glob('*.jpg')) + list(label_dir.glob('*.png'))
            print(f"Processing {len(image_files)} images from {label_name} in {data_path.name}...")
            
            for img_path in image_files:
                try:
                    features = extract_roi_features(img_path)
                    features['label'] = label
                    features['binary_label'] = binary_labels_map[label_name]
                    features['image_path'] = str(img_path)
                    features['class_name'] = label_name
                    features_list.append(features)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(features_list, f, indent=2)
    
    print(f"Extracted features from {len(features_list)} images")
    print(f"Saved to {output_path}")
    
    return features_list

if __name__ == "__main__":
    # Default data path - adjust as needed
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Try to find the dataset
        possible_paths = [
            Path.home() / ".cache/kagglehub/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/versions/1",
            Path("data/raw/alzheimers_dataset/Combined Dataset"),
        ]
        data_dir = None
        for path in possible_paths:
            if Path(path).exists():
                data_dir = path
                break
        
        if data_dir is None:
            print("Please provide data directory as argument:")
            print("python extract_roi_features.py <path_to_dataset>")
            sys.exit(1)
    
    extract_features_from_directory(data_dir, 'data/roi_features.json')

