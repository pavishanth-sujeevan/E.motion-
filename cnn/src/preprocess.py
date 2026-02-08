"""
Preprocessing script for RAVDESS and TESS datasets
Extracts features from audio files and prepares train/test splits
"""
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

import config


def extract_features(file_path, duration=config.DURATION, sr=config.SAMPLE_RATE):
    """
    Extract audio features from a file

    Args:
        file_path: Path to audio file
        duration: Duration to pad/truncate audio to
        sr: Sample rate

    Returns:
        Feature array combining MFCC, mel spectrogram, chroma, and spectral features
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
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.N_MFCC,
                                     hop_length=config.HOP_LENGTH, n_fft=config.N_FFT)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # 2. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=config.N_MELS,
                                             hop_length=config.HOP_LENGTH, n_fft=config.N_FFT)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)

        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=config.HOP_LENGTH)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # 4. Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=config.HOP_LENGTH)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            mel_mean, mel_std,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            [zcr_mean, zcr_std]
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def load_ravdess_data():
    """
    Load and extract features from RAVDESS dataset

    Returns:
        features: List of feature arrays
        labels: List of emotion labels
    """
    features = []
    labels = []

    print("Processing RAVDESS dataset...")

    if not os.path.exists(config.RAVDESS_PATH):
        print(f"RAVDESS path not found: {config.RAVDESS_PATH}")
        return [], []

    # Iterate through actor folders
    actor_folders = [f for f in os.listdir(config.RAVDESS_PATH)
                     if f.startswith('Actor_') and os.path.isdir(os.path.join(config.RAVDESS_PATH, f))]

    for actor_folder in tqdm(actor_folders, desc="RAVDESS actors"):
        actor_path = os.path.join(config.RAVDESS_PATH, actor_folder)

        # Get all .wav files
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

        for audio_file in audio_files:
            # RAVDESS filename format: 03-01-06-01-02-01-12.wav
            # Position 3 (index 2) is the emotion
            parts = audio_file.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = config.RAVDESS_EMOTIONS.get(emotion_code)

                # Only process emotions in our target list
                if emotion in config.EMOTIONS:
                    file_path = os.path.join(actor_path, audio_file)
                    feature = extract_features(file_path)

                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)

    print(f"Loaded {len(features)} samples from RAVDESS")
    return features, labels


def load_tess_data():
    """
    Load and extract features from TESS dataset

    Returns:
        features: List of feature arrays
        labels: List of emotion labels
    """
    features = []
    labels = []

    print("Processing TESS dataset...")

    if not os.path.exists(config.TESS_PATH):
        print(f"TESS path not found: {config.TESS_PATH}")
        return [], []

    # Get all emotion folders
    emotion_folders = [f for f in os.listdir(config.TESS_PATH)
                       if os.path.isdir(os.path.join(config.TESS_PATH, f))]

    for emotion_folder in tqdm(emotion_folders, desc="TESS emotions"):
        # Extract emotion from folder name (e.g., "OAF_angry" -> "angry")
        emotion = emotion_folder.split('_')[-1].lower()

        # Only process emotions in our target list
        if emotion in config.EMOTIONS:
            emotion_path = os.path.join(config.TESS_PATH, emotion_folder)

            # Get all .wav files
            audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]

            for audio_file in audio_files:
                file_path = os.path.join(emotion_path, audio_file)
                feature = extract_features(file_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)

    print(f"Loaded {len(features)} samples from TESS")
    return features, labels


def preprocess_and_save():
    """
    Main preprocessing function that loads both datasets,
    combines them, and saves train/test splits
    """
    print("Starting preprocessing...")
    print(f"Target emotions: {config.EMOTIONS}\n")

    # Load both datasets
    ravdess_features, ravdess_labels = load_ravdess_data()
    tess_features, tess_labels = load_tess_data()

    # Combine datasets
    all_features = ravdess_features + tess_features
    all_labels = ravdess_labels + tess_labels

    if len(all_features) == 0:
        print("No data loaded! Please check your dataset paths.")
        return

    print(f"\nTotal samples: {len(all_features)}")

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    # Print dataset statistics
    print("\nDataset statistics:")
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"  {emotion}: {count} samples")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label encoder classes for later use
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'),
            label_encoder.classes_)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded
    )

    # Normalize features (using training data statistics)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Save normalization parameters
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'feature_mean.npy'), mean)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'feature_std.npy'), std)

    # Save processed data
    print("\nSaving processed data...")
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

    print(f"\nPreprocessing complete!")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"\nProcessed data saved to: {config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    preprocess_and_save()