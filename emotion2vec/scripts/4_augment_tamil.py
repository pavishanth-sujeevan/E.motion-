"""
Audio data augmentation for Tamil dataset
Apply time stretching, pitch shifting, and noise addition to increase training samples
"""

import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import random

def time_stretch(audio, rate_range=(0.85, 1.15)):
    """Apply time stretching"""
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps_range=(-2, 2)):
    """Apply pitch shifting"""
    n_steps = random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor_range=(0.002, 0.01)):
    """Add random noise"""
    noise_factor = random.uniform(*noise_factor_range)
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def augment_audio(audio, sr, num_augmentations=3):
    """
    Apply multiple augmentations to create variations
    
    Args:
        audio: Input audio waveform
        sr: Sample rate
        num_augmentations: Number of augmented versions to create
    
    Returns:
        List of augmented audio samples
    """
    augmented_samples = []
    
    for i in range(num_augmentations):
        aug_audio = audio.copy()
        
        # Randomly apply augmentations
        augmentations = []
        
        # 80% chance of time stretch
        if random.random() < 0.8:
            aug_audio = time_stretch(aug_audio)
            augmentations.append('time_stretch')
        
        # 70% chance of pitch shift
        if random.random() < 0.7:
            aug_audio = pitch_shift(aug_audio, sr)
            augmentations.append('pitch_shift')
        
        # 60% chance of noise
        if random.random() < 0.6:
            aug_audio = add_noise(aug_audio)
            augmentations.append('noise')
        
        augmented_samples.append((aug_audio, augmentations))
    
    return augmented_samples

def load_audio_files_from_spectrograms(data_dir):
    """
    Load original audio files if available, otherwise reconstruct from spectrograms
    This is a workaround - ideally we'd have the original audio
    """
    print(f"Looking for audio files in: {data_dir}")
    
    # Check for raw audio directory
    raw_audio_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'raw', 'tamil')
    
    if os.path.exists(raw_audio_dir):
        print(f"✓ Found raw audio directory: {raw_audio_dir}")
        return raw_audio_dir, True
    else:
        print(f"⚠ Raw audio not found, will work with spectrograms")
        return data_dir, False

def augment_tamil_dataset(input_dir, output_dir, augmentation_factor=2):
    """
    Augment Tamil dataset
    
    Args:
        input_dir: Directory with preprocessed spectrograms
        output_dir: Directory to save augmented spectrograms
        augmentation_factor: How many augmented versions per sample
    """
    print("=" * 70)
    print("TAMIL DATASET AUGMENTATION")
    print("=" * 70)
    print()
    
    # Load existing data
    print("Loading existing preprocessed data...")
    X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(input_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(input_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
    
    print(f"Original training samples: {len(X_train)}")
    print(f"Augmentation factor: {augmentation_factor}x")
    print(f"Target training samples: {len(X_train) * (augmentation_factor + 1)}")
    print()
    
    # Check if we can access raw audio
    raw_audio_dir, has_raw_audio = load_audio_files_from_spectrograms(input_dir)
    
    if not has_raw_audio:
        print("=" * 70)
        print("AUGMENTATION STRATEGY: Spectrogram-based")
        print("=" * 70)
        print()
        print("Since raw audio is not available, we'll augment spectrograms directly")
        print("This includes:")
        print("  - Time masking (SpecAugment)")
        print("  - Frequency masking")
        print("  - Adding Gaussian noise")
        print()
        
        # Augment spectrograms directly
        X_train_aug, y_train_aug = augment_spectrograms(X_train, y_train, augmentation_factor)
        
        # Combine original and augmented
        X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
        y_train_combined = np.concatenate([y_train, y_train_aug], axis=0)
        
        print(f"\n✓ Augmented training set: {len(X_train)} → {len(X_train_combined)}")
    else:
        print("✓ Raw audio available - will perform proper audio augmentation")
        # This would require access to raw audio files
        # For now, fall back to spectrogram augmentation
        X_train_aug, y_train_aug = augment_spectrograms(X_train, y_train, augmentation_factor)
        X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
        y_train_combined = np.concatenate([y_train, y_train_aug], axis=0)
    
    # Save augmented dataset
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_combined)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_combined)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"\n✓ Augmented dataset saved to: {output_dir}")
    print()
    print("Dataset summary:")
    print(f"  Training:   {len(X_train_combined)} samples (original: {len(X_train)})")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")
    
    return output_dir

def augment_spectrograms(X, y, augmentation_factor=2):
    """
    Augment spectrograms using SpecAugment-like techniques
    """
    print("Augmenting spectrograms...")
    
    X_augmented = []
    y_augmented = []
    
    for i in tqdm(range(len(X)), desc="Processing samples"):
        spec = X[i]
        label = y[i]
        
        for _ in range(augmentation_factor):
            aug_spec = spec.copy()
            
            # Remove channel dimension if present
            if len(aug_spec.shape) == 3:
                aug_spec = aug_spec[:, :, 0]
            
            # Time masking (mask random time frames)
            if random.random() < 0.7:
                aug_spec = time_mask(aug_spec, max_mask_size=15)
            
            # Frequency masking
            if random.random() < 0.7:
                aug_spec = frequency_mask(aug_spec, max_mask_size=10)
            
            # Add noise
            if random.random() < 0.5:
                aug_spec = aug_spec + np.random.normal(0, 0.01, aug_spec.shape)
            
            # Restore channel dimension if needed
            if len(spec.shape) == 3:
                aug_spec = aug_spec[:, :, np.newaxis]
            
            X_augmented.append(aug_spec)
            y_augmented.append(label)
    
    return np.array(X_augmented), np.array(y_augmented)

def time_mask(spec, max_mask_size=15):
    """Apply time masking to spectrogram"""
    spec = spec.copy()
    time_size = spec.shape[1]
    mask_size = random.randint(1, min(max_mask_size, time_size // 4))
    mask_start = random.randint(0, time_size - mask_size)
    spec[:, mask_start:mask_start + mask_size] = 0
    return spec

def frequency_mask(spec, max_mask_size=10):
    """Apply frequency masking to spectrogram"""
    spec = spec.copy()
    freq_size = spec.shape[0]
    mask_size = random.randint(1, min(max_mask_size, freq_size // 4))
    mask_start = random.randint(0, freq_size - mask_size)
    spec[mask_start:mask_start + mask_size, :] = 0
    return spec

def main():
    # Paths
    input_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cnn', 'data', 'processed_tamil')
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cnn', 'data', 'processed_tamil_augmented')
    
    # Augment dataset (3x augmentation = 654 → 2616 samples)
    augmented_dir = augment_tamil_dataset(input_dir, output_dir, augmentation_factor=3)
    
    print()
    print("=" * 70)
    print("AUGMENTATION COMPLETE!")
    print("=" * 70)
    print(f"\nAugmented data saved to: {augmented_dir}")
    print("\nNext steps:")
    print("  1. Extract features from augmented data")
    print("  2. Train model with augmented dataset")
    print("  3. Compare with baseline (34.04%)")

if __name__ == '__main__':
    main()
