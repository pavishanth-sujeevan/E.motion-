"""
IMPROVED Preprocessing script with better class balancing and data inspection
"""
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config


def extract_features(file_path, duration=config.DURATION, sr=config.SAMPLE_RATE):
    """
    Extract audio features from a file with improved feature extraction
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)

        # Pad or truncate to fixed length
        max_len = sr * duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]

        # Extract features
        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.N_MFCC,
                                     hop_length=config.HOP_LENGTH, n_fft=config.N_FFT)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_max = np.max(mfccs, axis=1)
        mfccs_min = np.min(mfccs, axis=1)

        # 2. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40,
                                            hop_length=config.HOP_LENGTH, n_fft=config.N_FFT)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)

        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=config.HOP_LENGTH)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=config.HOP_LENGTH)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=config.HOP_LENGTH)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=config.HOP_LENGTH)

        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH)

        # 6. RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)

        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std, mfccs_max, mfccs_min,
            mel_mean, mel_std,
            chroma_mean, chroma_std,
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)]
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def load_ravdess_data():
    """Load RAVDESS dataset"""
    features = []
    labels = []
    file_paths = []

    print("Processing RAVDESS dataset...")

    if not os.path.exists(config.RAVDESS_PATH):
        print(f"RAVDESS path not found: {config.RAVDESS_PATH}")
        return [], [], []

    actor_folders = [f for f in os.listdir(config.RAVDESS_PATH)
                    if f.startswith('Actor_') and os.path.isdir(os.path.join(config.RAVDESS_PATH, f))]

    for actor_folder in tqdm(actor_folders, desc="RAVDESS actors"):
        actor_path = os.path.join(config.RAVDESS_PATH, actor_folder)
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

        for audio_file in audio_files:
            parts = audio_file.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = config.RAVDESS_EMOTIONS.get(emotion_code)

                if emotion in config.EMOTIONS:
                    file_path = os.path.join(actor_path, audio_file)
                    feature = extract_features(file_path)

                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)
                        file_paths.append(file_path)

    print(f"Loaded {len(features)} samples from RAVDESS")
    return features, labels, file_paths


def load_tess_data():
    """Load TESS dataset"""
    features = []
    labels = []
    file_paths = []

    print("Processing TESS dataset...")

    if not os.path.exists(config.TESS_PATH):
        print(f"TESS path not found: {config.TESS_PATH}")
        return [], [], []

    emotion_folders = [f for f in os.listdir(config.TESS_PATH)
                      if os.path.isdir(os.path.join(config.TESS_PATH, f))]

    for emotion_folder in tqdm(emotion_folders, desc="TESS emotions"):
        emotion = emotion_folder.split('_')[-1].lower()

        if emotion in config.EMOTIONS:
            emotion_path = os.path.join(config.TESS_PATH, emotion_folder)
            audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]

            for audio_file in audio_files:
                file_path = os.path.join(emotion_path, audio_file)
                feature = extract_features(file_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
                    file_paths.append(file_path)

    print(f"Loaded {len(features)} samples from TESS")
    return features, labels, file_paths


def balance_classes(X, y, file_paths):
    """Balance classes using random oversampling"""
    print("\nBalancing classes...")

    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts)

    print("Original distribution:")
    for emotion, count in zip(unique, counts):
        print(f"  {emotion}: {count}")

    # Combine data
    data = list(zip(X, y, file_paths))

    # Separate by class
    class_data = {emotion: [] for emotion in unique}
    for x, label, path in data:
        class_data[label].append((x, label, path))

    # Oversample minority classes
    balanced_data = []
    for emotion in unique:
        emotion_data = class_data[emotion]
        if len(emotion_data) < max_count:
            # Oversample
            emotion_data = resample(emotion_data,
                                   n_samples=max_count,
                                   random_state=config.RANDOM_STATE,
                                   replace=True)
        balanced_data.extend(emotion_data)

    # Unpack
    X_balanced = [item[0] for item in balanced_data]
    y_balanced = [item[1] for item in balanced_data]
    paths_balanced = [item[2] for item in balanced_data]

    print("\nBalanced distribution:")
    unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
    for emotion, count in zip(unique_balanced, counts_balanced):
        print(f"  {emotion}: {count}")

    return X_balanced, y_balanced, paths_balanced


def preprocess_and_save(balance_data=True):
    """
    Main preprocessing function

    Args:
        balance_data: Whether to balance classes (recommended)
    """
    print("Starting preprocessing...")
    print(f"Target emotions: {config.EMOTIONS}\n")

    # Load both datasets
    ravdess_features, ravdess_labels, ravdess_paths = load_ravdess_data()
    tess_features, tess_labels, tess_paths = load_tess_data()

    # Combine datasets
    all_features = ravdess_features + tess_features
    all_labels = ravdess_labels + tess_labels
    all_paths = ravdess_paths + tess_paths

    if len(all_features) == 0:
        print("No data loaded! Please check your dataset paths.")
        return

    print(f"\nTotal samples: {len(all_features)}")

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    # Print dataset statistics
    print("\n" + "="*60)
    print("ORIGINAL DATASET STATISTICS")
    print("="*60)
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  {emotion:12s}: {count:4d} samples ({percentage:5.1f}%)")

    # Balance classes if requested
    if balance_data:
        all_features, all_labels, all_paths = balance_classes(
            all_features, all_labels, all_paths
        )
        X = np.array(all_features)
        y = np.array(all_labels)

        print(f"\nTotal samples after balancing: {len(X)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label encoder classes
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'),
            label_encoder.classes_)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded
    )

    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT")
    print("="*60)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Check class distribution in splits
    print("\nTraining set distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for idx, count in zip(unique_train, counts_train):
        emotion = label_encoder.classes_[idx]
        percentage = (count / len(y_train)) * 100
        print(f"  {emotion:12s}: {count:4d} samples ({percentage:5.1f}%)")

    print("\nTest set distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for idx, count in zip(unique_test, counts_test):
        emotion = label_encoder.classes_[idx]
        percentage = (count / len(y_test)) * 100
        print(f"  {emotion:12s}: {count:4d} samples ({percentage:5.1f}%)")

    # Normalize features using StandardScaler approach
    print("\n" + "="*60)
    print("FEATURE NORMALIZATION")
    print("="*60)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Prevent division by zero
    std = np.where(std == 0, 1, std)

    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"Normalized train range: [{X_train_normalized.min():.4f}, {X_train_normalized.max():.4f}]")
    print(f"Normalized test range: [{X_test_normalized.min():.4f}, {X_test_normalized.max():.4f}]")

    # Save normalization parameters
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'feature_mean.npy'), mean)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'feature_std.npy'), std)

    # Save processed data
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)

    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train_normalized)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test_normalized)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

    print(f"✓ Processed data saved to: {config.PROCESSED_DATA_DIR}")
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nNext step: Run training")
    print("  python src/train.py")


if __name__ == "__main__":
    preprocess_and_save(balance_data=True)