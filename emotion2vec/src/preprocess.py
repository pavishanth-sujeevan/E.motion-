import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# Paths based on your project structure images
BASE_DIR = Path(r"F:\Stage 2\CM2603 - DSGP\ser\emotion2vec")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

# RAVDESS Emotion Mapping from your identifier image
RAVDESS_EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}


def clean_audio(input_path, output_name):
    """Standardize audio for emotion2vec."""
    try:
        y, _ = librosa.load(input_path, sr=TARGET_SR)
        y, _ = librosa.effects.trim(y, top_db=20)  # Remove leading/trailing silence
        out_path = PROCESSED_DIR / output_name
        sf.write(out_path, y, TARGET_SR)
        return str(out_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def run_preprocessing():
    manifest = []

    # 1. Process RAVDESS-SPEECH
    ravdess_path = RAW_DIR / "RAVDESS-SPEECH"
    print("Processing RAVDESS...")
    # Search recursively for .wav files in Actor_XX folders
    for path in tqdm(list(ravdess_path.rglob('*.wav'))):
        parts = path.name.split('-')
        # Check Modality (03 = audio-only) and Vocal Channel (01 = speech)
        if parts[0] == "03" and parts[1] == "01":
            emotion = RAVDESS_EMOTION_MAP.get(parts[2])
            new_name = f"ravdess_{path.name}"
            final_path = clean_audio(path, new_name)
            if final_path:
                manifest.append({"path": final_path, "label": emotion, "dataset": "ravdess"})

    # 2. Process TESS
    tess_path = RAW_DIR / "TESS" / "TESS Toronto emotional speech set data"
    print("Processing TESS...")
    # TESS files look like 'OAF_back_happy.wav' inside 'OAF_happy' folders
    for path in tqdm(list(tess_path.rglob('*.wav'))):
        # Extract emotion from filename: 'happy' from 'OAF_back_happy.wav'
        emotion_tag = path.stem.split('_')[-1].lower()

        # Standardize labels (TESS uses 'fear', RAVDESS uses 'fearful')
        if emotion_tag == 'ps':
            emotion = 'surprised'
        elif emotion_tag == 'fear':
            emotion = 'fearful'
        else:
            emotion = emotion_tag

        new_name = f"tess_{path.name}"
        final_path = clean_audio(path, new_name)
        if final_path:
            manifest.append({"path": final_path, "label": emotion, "dataset": "tess"})

    # Save metadata to the project root
    df = pd.DataFrame(manifest)
    output_csv = BASE_DIR / "metadata.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Saved {len(df)} records to {output_csv}")


if __name__ == "__main__":
    run_preprocessing()