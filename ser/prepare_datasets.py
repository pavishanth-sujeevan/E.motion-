"""
Prepare dataset CSV files for multilingual SER training
Creates CSV files with path and label columns from existing audio datasets
"""

import os
import pandas as pd
from pathlib import Path
import random

# Emotion label mapping (consistent across datasets)
EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6
}

def process_ravdess(base_path):
    """Process RAVDESS dataset"""
    print("Processing RAVDESS...")
    data = []
    
    ravdess_path = Path(base_path) / "RAVDESS-SPEECH"
    
    if not ravdess_path.exists():
        print(f"Warning: RAVDESS path not found: {ravdess_path}")
        return pd.DataFrame(columns=['path', 'label'])
    
    # RAVDESS emotion mapping: 01=neutral, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
    ravdess_emotions = {
        '01': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6
    }
    
    for actor_folder in ravdess_path.glob("Actor_*"):
        if actor_folder.is_dir():
            for audio_file in actor_folder.glob("*.wav"):
                # RAVDESS filename format: 03-01-06-01-02-01-12.wav
                # Position 3 is emotion
                parts = audio_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in ravdess_emotions:
                        data.append({
                            'path': str(audio_file.absolute()),
                            'label': ravdess_emotions[emotion_code]
                        })
    
    print(f"  Found {len(data)} RAVDESS samples")
    return pd.DataFrame(data)


def process_tess(base_path):
    """Process TESS dataset"""
    print("Processing TESS...")
    data = []
    
    tess_path = Path(base_path) / "TESS"
    
    if not tess_path.exists():
        print(f"Warning: TESS path not found: {tess_path}")
        return pd.DataFrame(columns=['path', 'label'])
    
    # TESS emotion mapping from folder names
    tess_emotions = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'disgust': 5,
        'ps': 6  # pleasant surprise
    }
    
    # TESS has nested structure, search recursively
    for emotion_folder in tess_path.rglob("*"):
        if emotion_folder.is_dir():
            emotion_name = emotion_folder.name.lower()
            
            # Extract emotion from folder name (e.g., "OAF_angry", "YAF_sad")
            for key in tess_emotions:
                if key in emotion_name:
                    label = tess_emotions[key]
                    for audio_file in emotion_folder.glob("*.wav"):
                        data.append({
                            'path': str(audio_file.absolute()),
                            'label': label
                        })
                    break
    
    print(f"  Found {len(data)} TESS samples")
    return pd.DataFrame(data)


def process_emota(base_path):
    """Process EmoTa (Tamil) dataset"""
    print("Processing EmoTa (Tamil)...")
    data = []
    
    emota_path = Path(base_path) / "EmoTa"
    
    if not emota_path.exists():
        print(f"Warning: EmoTa path not found: {emota_path}")
        return pd.DataFrame(columns=['path', 'label'])
    
    # EmoTa emotion mapping from filename suffixes
    emota_emotions = {
        'neu': 0,  # neutral
        'hap': 1,  # happy
        'sad': 2,  # sad
        'ang': 3,  # angry
        'fea': 4,  # fear
        'dis': 5   # disgust
    }
    
    # Search recursively for .wav files
    for audio_file in emota_path.rglob("*.wav"):
        filename = audio_file.stem.lower()
        
        # Extract emotion from filename (e.g., "01_01_ang.wav" -> "ang")
        for emotion_code, label in emota_emotions.items():
            if emotion_code in filename:
                data.append({
                    'path': str(audio_file.absolute()),
                    'label': label
                })
                break
    
    print(f"  Found {len(data)} EmoTa samples")
    return pd.DataFrame(data)


def create_sinhala_mock_data(base_path, num_labeled=100, num_unlabeled=500):
    """
    Create mock Sinhala dataset from existing audio files
    Since we don't have actual Sinhala data, we'll use a subset of English data as placeholder
    """
    print(f"Creating mock Sinhala data ({num_labeled} labeled, {num_unlabeled} unlabeled)...")
    
    # Use TESS as source for mock Sinhala data
    tess_path = Path(base_path) / "TESS"
    
    if not tess_path.exists():
        print("Warning: Cannot create mock Sinhala data - TESS not found")
        return pd.DataFrame(columns=['path', 'label']), pd.DataFrame(columns=['path'])
    
    # Collect all TESS files recursively
    all_files = []
    tess_emotions = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fear': 4, 'disgust': 5, 'ps': 6}
    
    for emotion_folder in tess_path.rglob("*"):
        if emotion_folder.is_dir():
            emotion_name = emotion_folder.name.lower()
            for key in tess_emotions:
                if key in emotion_name:
                    label = tess_emotions[key]
                    for audio_file in emotion_folder.glob("*.wav"):
                        all_files.append({'path': str(audio_file.absolute()), 'label': label})
                    break
    
    if len(all_files) < num_labeled + num_unlabeled:
        print(f"  Warning: Not enough files. Adjusting to available: {len(all_files)}")
        num_labeled = min(num_labeled, len(all_files) // 2)
        num_unlabeled = min(num_unlabeled, len(all_files) - num_labeled)
    
    # Shuffle and split
    random.shuffle(all_files)
    
    labeled_data = all_files[:num_labeled]
    unlabeled_data = [{'path': item['path']} for item in all_files[num_labeled:num_labeled + num_unlabeled]]
    
    print(f"  Created {len(labeled_data)} labeled and {len(unlabeled_data)} unlabeled samples")
    
    return pd.DataFrame(labeled_data), pd.DataFrame(unlabeled_data)


def main():
    """Main function to prepare all datasets"""
    
    # Base paths
    base_raw_path = Path("E:/Projects/E.motion-/emotion2vec/data/raw")
    output_dir = Path("E:/Projects/E.motion-/data")
    
    # Create output directory structure
    output_dir.mkdir(exist_ok=True)
    (output_dir / "english").mkdir(exist_ok=True)
    (output_dir / "tamil").mkdir(exist_ok=True)
    (output_dir / "sinhala").mkdir(exist_ok=True)
    
    print("="*60)
    print("Preparing Multilingual SER Datasets")
    print("="*60)
    print()
    
    # Process English (RAVDESS + TESS)
    print("Processing English datasets...")
    ravdess_df = process_ravdess(base_raw_path)
    tess_df = process_tess(base_raw_path)
    english_df = pd.concat([ravdess_df, tess_df], ignore_index=True)
    
    if len(english_df) > 0:
        english_df = english_df.sample(frac=1, random_state=42).reset_index(drop=True)
        english_path = output_dir / "english" / "labels.csv"
        english_df.to_csv(english_path, index=False)
        print(f"✓ Saved {len(english_df)} English samples to {english_path}")
    else:
        print("✗ No English data found")
    
    print()
    
    # Process Tamil (EmoTa)
    print("Processing Tamil dataset...")
    tamil_df = process_emota(base_raw_path)
    
    if len(tamil_df) > 0:
        tamil_df = tamil_df.sample(frac=1, random_state=42).reset_index(drop=True)
        tamil_path = output_dir / "tamil" / "labels.csv"
        tamil_df.to_csv(tamil_path, index=False)
        print(f"✓ Saved {len(tamil_df)} Tamil samples to {tamil_path}")
    else:
        print("✗ No Tamil data found")
    
    print()
    
    # Create mock Sinhala data
    print("Processing Sinhala dataset...")
    sinhala_labeled_df, sinhala_unlabeled_df = create_sinhala_mock_data(base_raw_path)
    
    if len(sinhala_labeled_df) > 0:
        labeled_path = output_dir / "sinhala" / "labeled.csv"
        unlabeled_path = output_dir / "sinhala" / "unlabeled.csv"
        
        sinhala_labeled_df.to_csv(labeled_path, index=False)
        sinhala_unlabeled_df.to_csv(unlabeled_path, index=False)
        
        print(f"✓ Saved {len(sinhala_labeled_df)} labeled Sinhala samples to {labeled_path}")
        print(f"✓ Saved {len(sinhala_unlabeled_df)} unlabeled Sinhala samples to {unlabeled_path}")
    else:
        print("✗ No Sinhala data created")
    
    print()
    print("="*60)
    print("Dataset Preparation Complete!")
    print("="*60)
    print()
    print("Summary:")
    print(f"  English: {len(english_df)} samples")
    print(f"  Tamil: {len(tamil_df)} samples")
    print(f"  Sinhala: {len(sinhala_labeled_df)} labeled + {len(sinhala_unlabeled_df)} unlabeled")
    print()
    print("Output directory: E:/Projects/E.motion-/data/")
    print()


if __name__ == "__main__":
    main()
