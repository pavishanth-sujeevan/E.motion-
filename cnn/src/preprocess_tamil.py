"""
Preprocessing for Tamil-only model
Uses only EMOTA dataset
"""
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config


def extract_melspectrogram(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION):
    """Extract mel spectrogram from audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        max_len = sr * duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=config.N_MELS,
            n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        if mel_spec_normalized.shape[1] < config.MAX_TIME_STEPS:
            pad_width = config.MAX_TIME_STEPS - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :config.MAX_TIME_STEPS]

        return mel_spec_normalized

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def load_tamil_data():
    """Load Tamil (EMOTA) dataset only"""
    spectrograms = []
    labels = []
    file_info = []

    print("Processing Tamil (EMOTA) dataset...")

    if not os.path.exists(config.EMOTA_PATH):
        print(f"EMOTA path not found: {config.EMOTA_PATH}")
        return [], [], []

    emota_emotion_map = {
        'ang': 'angry',
        'fea': 'fear',
        'hap': 'happy',
        'neu': 'neutral',
        'sad': 'sad'
    }

    audio_files = sorted([f for f in os.listdir(config.EMOTA_PATH) if f.endswith('.wav')])
    print(f"Found {len(audio_files)} audio files")

    for audio_file in tqdm(audio_files, desc="Tamil-EMOTA"):
        parts = audio_file.replace('.wav', '').split('_')
        
        if len(parts) >= 3:
            emotion_code = parts[-1]
            emotion = emota_emotion_map.get(emotion_code)
            
            if emotion and emotion in config.EMOTIONS:
                file_path = os.path.join(config.EMOTA_PATH, audio_file)
                mel_spec = extract_melspectrogram(file_path)
                
                if mel_spec is not None:
                    spectrograms.append(mel_spec)
                    labels.append(emotion)
                    file_info.append({
                        'file': audio_file,
                        'speaker': parts[0],
                        'utterance': parts[1],
                        'emotion': emotion,
                        'dataset': 'EMOTA'
                    })

    print(f"Loaded {len(spectrograms)} samples from Tamil EMOTA")
    return spectrograms, labels, file_info


def preprocess_tamil_only():
    """Preprocess Tamil data only"""
    print("="*70)
    print("TAMIL-ONLY MEL SPECTROGRAM PREPROCESSING")
    print("="*70)
    print(f"\nTarget emotions: {config.EMOTIONS}")
    print(f"Spectrogram shape: ({config.N_MELS}, {config.MAX_TIME_STEPS})")
    print(f"Duration: {config.DURATION} seconds")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz\n")

    # Load Tamil data only
    tamil_specs, tamil_labels, tamil_info = load_tamil_data()

    if len(tamil_specs) == 0:
        print("\n❌ No Tamil data loaded! Check dataset path.")
        return

    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples: {len(tamil_specs)}")

    # Class distribution
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*70}")

    from collections import Counter
    label_counts = Counter(tamil_labels)

    for emotion in config.EMOTIONS:
        count = label_counts.get(emotion, 0)
        percentage = (count / len(tamil_labels)) * 100 if tamil_labels else 0
        print(f"{emotion:12s}: {count:4d} samples ({percentage:5.1f}%)")

    # Convert to numpy arrays
    X = np.array(tamil_specs)
    y = np.array(tamil_labels)

    # Add channel dimension for CNN
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

    print(f"Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X))*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({(X_val.shape[0]/len(X))*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({(X_test.shape[0]/len(X))*100:.1f}%)")

    # Print class distributions
    for split_name, split_data in [('Training', y_train), ('Validation', y_val), ('Test', y_test)]:
        print(f"\n{split_name} set distribution:")
        counts = np.bincount(split_data, minlength=len(config.EMOTIONS))
        for idx, count in enumerate(counts):
            emotion = config.INDEX_TO_EMOTION[idx]
            percentage = (count / len(split_data)) * 100
            print(f"  {emotion:12s}: {count:4d} ({percentage:5.1f}%)")

    # Save processed data
    print(f"\n{'='*70}")
    print("SAVING PROCESSED DATA")
    print(f"{'='*70}")

    # Create Tamil-specific processed data directory
    tamil_processed_dir = os.path.join(config.DATA_DIR, 'processed_tamil')
    os.makedirs(tamil_processed_dir, exist_ok=True)

    np.save(os.path.join(tamil_processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(tamil_processed_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(tamil_processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(tamil_processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(tamil_processed_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(tamil_processed_dir, 'y_test.npy'), y_test)

    print(f"[OK] Tamil data saved to: {tamil_processed_dir}")

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext step: Train the Tamil model")
    print(f"  python train_tamil.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    preprocess_tamil_only()
