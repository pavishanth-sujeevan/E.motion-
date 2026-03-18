import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Paths ---
BASE_DIR = Path(r"F:\Stage 2\CM2603 - DSGP\ser\emotion2vec")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
TEST_SIZE_RECORDS = 20  # <--- Change this number to reserve more/less for testing

RAVDESS_EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}


def clean_audio(input_path, output_name):
    """Standardize audio for emotion2vec."""
    try:
        y, _ = librosa.load(input_path, sr=TARGET_SR)
        y, _ = librosa.effects.trim(y, top_db=20)
        out_path = PROCESSED_DIR / output_name
        sf.write(out_path, y, TARGET_SR)
        return str(out_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def run_preprocessing():
    manifest = []

    # 1. Process RAVDESS
    ravdess_path = RAW_DIR / "RAVDESS-SPEECH"
    print("Gathering RAVDESS files...")
    for path in tqdm(list(ravdess_path.rglob('*.wav'))):
        parts = path.name.split('-')
        if parts[0] == "03" and parts[1] == "01":
            emotion = RAVDESS_EMOTION_MAP.get(parts[2])
            manifest.append({"raw_path": path, "new_name": f"ravdess_{path.name}", "label": emotion})

    # 2. Process TESS
    tess_path = RAW_DIR / "TESS" / "TESS Toronto emotional speech set data"
    print("Gathering TESS files...")
    for path in tqdm(list(tess_path.rglob('*.wav'))):
        emotion_tag = path.stem.split('_')[-1].lower()
        if emotion_tag == 'ps':
            emotion = 'surprised'
        elif emotion_tag == 'fear':
            emotion = 'fearful'
        else:
            emotion = emotion_tag
        manifest.append({"raw_path": path, "new_name": f"tess_{path.name}", "label": emotion})

    # --- THE SPLIT LOGIC ---
    df_all = pd.DataFrame(manifest)

    # Stratified split ensures your 20 test records have a mix of all emotions
    # Using 'test_size=TEST_SIZE_RECORDS' directly since it's an integer
    train_df, test_df = train_test_split(
        df_all,
        test_size=TEST_SIZE_RECORDS,
        stratify=df_all['label'],
        random_state=42
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, test_df])

    # --- PHYSICAL PROCESSING ---
    print(f"Cleaning and saving {len(final_df)} audio files...")
    processed_paths = []
    for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
        p_path = clean_audio(row['raw_path'], row['new_name'])
        processed_paths.append(p_path)

    final_df['path'] = processed_paths

    # Save clean metadata
    output_df = final_df[['path', 'label', 'split']]
    output_df.to_csv(BASE_DIR / "metadata.csv", index=False)
    print(f"\nSuccess! metadata.csv created with {len(train_df)} train and {len(test_df)} test records.")


if __name__ == "__main__":
    run_preprocessing()