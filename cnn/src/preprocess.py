"""
Preprocessing script using Mel Spectrograms
Converts audio files to mel spectrogram images
"""
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config


def extract_melspectrogram(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION):
    """
    Extract mel spectrogram from audio file

    Returns:
        mel_spectrogram: 2D numpy array of shape (n_mels, time_steps)
    """
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)

        # Ensure fixed length
        max_len = sr * duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]

        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # Ensure fixed time dimension
        if mel_spec_normalized.shape[1] < config.MAX_TIME_STEPS:
            # Pad
            pad_width = config.MAX_TIME_STEPS - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mel_spec_normalized = mel_spec_normalized[:, :config.MAX_TIME_STEPS]

        return mel_spec_normalized

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def visualize_sample_spectrograms(spectrograms, labels, save_path):
    """
    Visualize sample spectrograms from each emotion
    """
    emotions = config.EMOTIONS
    fig, axes = plt.subplots(1, len(emotions), figsize=(20, 4))

    for idx, emotion in enumerate(emotions):
        # Find first sample of this emotion
        emotion_indices = [i for i, label in enumerate(labels) if label == emotion]

        if emotion_indices:
            sample_idx = emotion_indices[0]
            spec = spectrograms[sample_idx]

            im = axes[idx].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[idx].set_title(f'{emotion.upper()}', fontweight='bold')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Mel Frequency')
            plt.colorbar(im, ax=axes[idx], format='%0.2f')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample spectrograms saved to: {save_path}")
    plt.close()


def load_ravdess_data():
    """Load RAVDESS dataset and extract mel spectrograms"""
    spectrograms = []
    labels = []
    file_info = []

    print("Processing RAVDESS dataset...")

    if not os.path.exists(config.RAVDESS_PATH):
        print(f"RAVDESS path not found: {config.RAVDESS_PATH}")
        return [], [], []

    actor_folders = sorted([f for f in os.listdir(config.RAVDESS_PATH)
                           if f.startswith('Actor_') and os.path.isdir(os.path.join(config.RAVDESS_PATH, f))])

    print(f"Found {len(actor_folders)} actors")

    for actor_folder in tqdm(actor_folders, desc="RAVDESS"):
        actor_path = os.path.join(config.RAVDESS_PATH, actor_folder)
        audio_files = sorted([f for f in os.listdir(actor_path) if f.endswith('.wav')])

        for audio_file in audio_files:
            # Parse filename: 03-01-06-01-02-01-12.wav
            # Position 2 (0-indexed) is emotion
            parts = audio_file.split('-')

            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = config.RAVDESS_EMOTIONS.get(emotion_code)

                if emotion in config.EMOTIONS:
                    file_path = os.path.join(actor_path, audio_file)
                    mel_spec = extract_melspectrogram(file_path)

                    if mel_spec is not None:
                        spectrograms.append(mel_spec)
                        labels.append(emotion)
                        file_info.append({
                            'file': audio_file,
                            'actor': actor_folder,
                            'emotion': emotion,
                            'dataset': 'RAVDESS'
                        })

    print(f"Loaded {len(spectrograms)} samples from RAVDESS")
    return spectrograms, labels, file_info


def load_tess_data():
    """Load TESS dataset and extract mel spectrograms"""
    spectrograms = []
    labels = []
    file_info = []

    print("Processing TESS dataset...")

    if not os.path.exists(config.TESS_PATH):
        print(f"TESS path not found: {config.TESS_PATH}")
        return [], [], []

    emotion_folders = sorted([f for f in os.listdir(config.TESS_PATH)
                             if os.path.isdir(os.path.join(config.TESS_PATH, f))])

    print(f"Found {len(emotion_folders)} emotion folders")

    for emotion_folder in tqdm(emotion_folders, desc="TESS"):
        # Extract emotion from folder name: OAF_angry -> angry
        parts = emotion_folder.split('_')
        if len(parts) >= 2:
            emotion = parts[1].lower()

            if emotion in config.EMOTIONS:
                emotion_path = os.path.join(config.TESS_PATH, emotion_folder)
                audio_files = sorted([f for f in os.listdir(emotion_path) if f.endswith('.wav')])

                for audio_file in audio_files:
                    file_path = os.path.join(emotion_path, audio_file)
                    mel_spec = extract_melspectrogram(file_path)

                    if mel_spec is not None:
                        spectrograms.append(mel_spec)
                        labels.append(emotion)
                        file_info.append({
                            'file': audio_file,
                            'folder': emotion_folder,
                            'emotion': emotion,
                            'dataset': 'TESS'
                        })

    print(f"Loaded {len(spectrograms)} samples from TESS")
    return spectrograms, labels, file_info


def preprocess_and_save():
    """Main preprocessing function"""
    print("="*70)
    print("MEL SPECTROGRAM PREPROCESSING")
    print("="*70)
    print(f"\nTarget emotions: {config.EMOTIONS}")
    print(f"Spectrogram shape: ({config.N_MELS}, {config.MAX_TIME_STEPS})")
    print(f"Duration: {config.DURATION} seconds")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz\n")

    # Load datasets
    ravdess_specs, ravdess_labels, ravdess_info = load_ravdess_data()
    tess_specs, tess_labels, tess_info = load_tess_data()

    # Combine
    all_spectrograms = ravdess_specs + tess_specs
    all_labels = ravdess_labels + tess_labels
    all_info = ravdess_info + tess_info

    if len(all_spectrograms) == 0:
        print("\n❌ No data loaded! Check dataset paths.")
        return

    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_spectrograms)}")
    print(f"\nRAVDESS: {len(ravdess_specs)} samples")
    print(f"TESS: {len(tess_specs)} samples")

    # Class distribution
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*70}")

    from collections import Counter
    label_counts = Counter(all_labels)

    for emotion in config.EMOTIONS:
        count = label_counts.get(emotion, 0)
        percentage = (count / len(all_labels)) * 100
        print(f"{emotion:12s}: {count:4d} samples ({percentage:5.1f}%)")

    # Convert to numpy arrays
    X = np.array(all_spectrograms)
    y = np.array(all_labels)

    # Add channel dimension for CNN (height, width, channels)
    X = X[..., np.newaxis]

    print(f"\nSpectrogram array shape: {X.shape}")
    print(f"Min value: {X.min():.4f}, Max value: {X.max():.4f}")

    # Encode labels to integers
    y_encoded = np.array([config.EMOTION_TO_INDEX[label] for label in y])

    # Split data
    print(f"\n{'='*70}")
    print("TRAIN/VALIDATION/TEST SPLIT")
    print(f"{'='*70}")

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded
    )

    # Second split: train vs val
    val_size = config.VALIDATION_SPLIT / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} samples ({(len(X_train)/len(X))*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({(len(X_val)/len(X))*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({(len(X_test)/len(X))*100:.1f}%)")

    # Check distribution in each set
    for split_name, y_split in [('Training', y_train), ('Validation', y_val), ('Test', y_test)]:
        print(f"\n{split_name} set distribution:")
        unique, counts = np.unique(y_split, return_counts=True)
        for idx, count in zip(unique, counts):
            emotion = config.INDEX_TO_EMOTION[idx]
            percentage = (count / len(y_split)) * 100
            print(f"  {emotion:12s}: {count:4d} ({percentage:5.1f}%)")

    # Save processed data
    print(f"\n{'='*70}")
    print("SAVING PROCESSED DATA")
    print(f"{'='*70}")

    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

    print(f"✓ Data saved to: {config.PROCESSED_DATA_DIR}")

    # Visualize sample spectrograms
    print("\nCreating sample visualizations...")
    viz_path = os.path.join(config.RESULTS_DIR, 'sample_spectrograms.png')
    visualize_sample_spectrograms(all_spectrograms[:100], all_labels[:100], viz_path)

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print("\nNext step: Train the model")
    print("  python train_spectrogram.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    preprocess_and_save()