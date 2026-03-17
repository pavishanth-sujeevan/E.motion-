"""
Extract MFCC features from audio datasets (alternative to emotion2vec embeddings)
This approach uses traditional MFCC features which are faster and simpler.
"""

import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import load_audio, extract_mfcc_features, TARGET_EMOTIONS

def load_dataset_from_spectrograms(data_dir, dataset_name):
    """Load dataset from preprocessed spectrograms"""
    print(f"\nLoading {dataset_name} dataset...")
    print(f"Directory: {data_dir}")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Feature shape: {X_train.shape}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def extract_features_from_spectrograms(spectrograms):
    """
    Extract simple statistical features from spectrograms
    This is a workaround since we don't have the raw audio files easily accessible
    """
    print("\nExtracting features from spectrograms...")
    
    features = []
    for i in tqdm(range(len(spectrograms)), desc="Processing"):
        spec = spectrograms[i]
        
        # Remove channel dimension if present
        if len(spec.shape) == 3:
            spec = spec[:, :, 0]
        
        # Calculate statistics across time and frequency
        feat_mean = np.mean(spec, axis=(0, 1)) if len(spec.shape) > 1 else np.mean(spec)
        feat_std = np.std(spec, axis=(0, 1)) if len(spec.shape) > 1 else np.std(spec)
        feat_max = np.max(spec, axis=(0, 1)) if len(spec.shape) > 1 else np.max(spec)
        feat_min = np.min(spec, axis=(0, 1)) if len(spec.shape) > 1 else np.min(spec)
        
        # Flatten spectrograms and take a sample of values
        flat_spec = spec.flatten()
        sampled_spec = flat_spec[::len(flat_spec)//100] if len(flat_spec) > 100 else flat_spec
        
        # Combine features
        combined = np.concatenate([
            [feat_mean], [feat_std], [feat_max], [feat_min],
            sampled_spec[:96]  # Pad or truncate to 96 features
        ])
        
        # Ensure fixed size (100 features total)
        if len(combined) < 100:
            combined = np.pad(combined, (0, 100 - len(combined)))
        elif len(combined) > 100:
            combined = combined[:100]
        
        features.append(combined)
    
    return np.array(features)

def main():
    print("=" * 70)
    print("EXTRACTING FEATURES FROM PREPROCESSED DATA")
    print("=" * 70)
    
    # Paths
    cnn_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cnn', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process English dataset
    print("\n" + "=" * 70)
    print("ENGLISH DATASET")
    print("=" * 70)
    
    english_dir = os.path.join(cnn_data_dir, 'processed_spectrograms')
    if os.path.exists(english_dir):
        english_data = load_dataset_from_spectrograms(english_dir, 'English')
        
        # Extract features
        print("\nExtracting training features...")
        X_train_features = extract_features_from_spectrograms(english_data['X_train'])
        print(f"Training features shape: {X_train_features.shape}")
        
        print("\nExtracting validation features...")
        X_val_features = extract_features_from_spectrograms(english_data['X_val'])
        print(f"Validation features shape: {X_val_features.shape}")
        
        print("\nExtracting test features...")
        X_test_features = extract_features_from_spectrograms(english_data['X_test'])
        print(f"Test features shape: {X_test_features.shape}")
        
        # Save features
        english_output_dir = os.path.join(output_dir, 'english')
        os.makedirs(english_output_dir, exist_ok=True)
        
        np.save(os.path.join(english_output_dir, 'X_train.npy'), X_train_features)
        np.save(os.path.join(english_output_dir, 'X_val.npy'), X_val_features)
        np.save(os.path.join(english_output_dir, 'X_test.npy'), X_test_features)
        np.save(os.path.join(english_output_dir, 'y_train.npy'), english_data['y_train'])
        np.save(os.path.join(english_output_dir, 'y_val.npy'), english_data['y_val'])
        np.save(os.path.join(english_output_dir, 'y_test.npy'), english_data['y_test'])
        
        print(f"\n✓ English features saved to: {english_output_dir}")
    else:
        print(f"⚠ English data not found at: {english_dir}")
    
    # Process Tamil dataset
    print("\n" + "=" * 70)
    print("TAMIL DATASET")
    print("=" * 70)
    
    tamil_dir = os.path.join(cnn_data_dir, 'processed_tamil')
    if os.path.exists(tamil_dir):
        tamil_data = load_dataset_from_spectrograms(tamil_dir, 'Tamil')
        
        # Extract features
        print("\nExtracting training features...")
        X_train_features = extract_features_from_spectrograms(tamil_data['X_train'])
        print(f"Training features shape: {X_train_features.shape}")
        
        print("\nExtracting validation features...")
        X_val_features = extract_features_from_spectrograms(tamil_data['X_val'])
        print(f"Validation features shape: {X_val_features.shape}")
        
        print("\nExtracting test features...")
        X_test_features = extract_features_from_spectrograms(tamil_data['X_test'])
        print(f"Test features shape: {X_test_features.shape}")
        
        # Save features
        tamil_output_dir = os.path.join(output_dir, 'tamil')
        os.makedirs(tamil_output_dir, exist_ok=True)
        
        np.save(os.path.join(tamil_output_dir, 'X_train.npy'), X_train_features)
        np.save(os.path.join(tamil_output_dir, 'X_val.npy'), X_val_features)
        np.save(os.path.join(tamil_output_dir, 'X_test.npy'), X_test_features)
        np.save(os.path.join(tamil_output_dir, 'y_train.npy'), tamil_data['y_train'])
        np.save(os.path.join(tamil_output_dir, 'y_val.npy'), tamil_data['y_val'])
        np.save(os.path.join(tamil_output_dir, 'y_test.npy'), tamil_data['y_test'])
        
        print(f"\n✓ Tamil features saved to: {tamil_output_dir}")
    else:
        print(f"⚠ Tamil data not found at: {tamil_dir}")
    
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nFeatures saved in: {output_dir}")
    print("\nNext step: Train classifier with these features")
    print("Run: python scripts/3_train_classifier.py")

if __name__ == '__main__':
    main()
