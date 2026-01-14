import re
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
import unicodedata

# ---------------------------
# Tamil character set
# ---------------------------
vowels = ['அ','ஆ','இ','ஈ','உ','ஊ','எ','ஏ','ஐ','ஒ','ஓ','ஔ']
vowel_strokes = ['ா','ி','ீ','ு','ூ','ெ','ே','ை','ொ','ோ','ௌ','்']
consonants = [
    'க','ங','ச','ஞ','ட','ண','த','ந','ப','ம','ய','ர','ல','வ','ழ','ள','ற','ன',
    'ஷ','ஜ','ஹ','க்ஷ'
]

tamil_chars = set(vowels + vowel_strokes + consonants)



# Paths
metadata_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\raw\\Tamil\\metadata.tsv"
audio_folder = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\raw\\Tamil\\wavs"

# Load dataset
df = pd.read_csv(metadata_file, sep="\t", names=["file_path","sentence"], encoding="utf-8")
df["file_path"] = df['file_path'].fillna('')
df['file_path'] = df['file_path'].apply(lambda x: os.path.join(audio_folder, os.path.basename(x.strip())))

# Basic info
print("Dataset shape: ", df.shape)
print("\nColumns: ", df.columns)
print("\nFirst 5 rows: \n", df.head())

# Check for missing values and duplicates
df = df.dropna(subset=["sentence","file_path"])
df = df[df["sentence"].str.strip().astype(bool)]
df = df.drop_duplicates(subset=["sentence","file_path"])


# Text normalization
def normalize_currency(text):
    text = re.sub(r"\bRs\.?\b", "ரூ", text)
    text = re.sub(r"\$", "டாலர்", text)
    text = re.sub(r"£", "பவுண்டு", text)
    return text


def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = normalize_currency(text)
    return text

# Apply cleaning
df['cleaned_text'] = df['sentence'].apply(clean_text)

# Save cleaned metadata
df_cleaned = pd.DataFrame({
    "audio_path": df['file_path'].apply(lambda x: os.path.split(x.strip())[-1]),
    "text": df["cleaned_text"]
})
df_cleaned = df_cleaned[['audio_path','text']]

output_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\Tamil\\tam_metadata_cleaned.tsv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_cleaned.to_csv(output_file, sep="|", index=False, header=False)
print("\nCleaned Tamil metadata saved successfully!")


# Audio preprocessing 
wav_dir = audio_folder
metadata_file = output_file
mel_dir = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\Tamil\\melspectrograms"
linear_dir = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\Tamil\\linear"
output_metadata_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\Tamil\\tam_ metadata_processed.tsv"

os.makedirs(mel_dir, exist_ok=True)
os.makedirs(linear_dir, exist_ok=True)

# Audio config
sample_rate = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 8000

# Mel spectrogram
def wav_to_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# Linear spectrogram
def wav_to_linear(y):
    D = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    magnitude = np.abs(D)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return magnitude_db

# Processing
metadata_output = []
with open(metadata_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Processing Tamil audio...")

for line in tqdm(lines):
    wav_id, text = line.strip().split("|")
    base_id = os.path.splitext(wav_id)[0]
    wav_path = os.path.join(wav_dir, wav_id)
    wav_path=unicodedata.normalize('NFC',wav_path)
    if not wav_path.endswith(".wav"):
        wav_path += ".wav"

    # Load audio
    y, sr = librosa.load(wav_path, sr=sample_rate)
    y, _ = librosa.effects.trim(y, top_db=20)  # trim silence
    y = y / (np.max(np.abs(y)) + 1e-9)          # normalize

    # Spectrograms
    mel = wav_to_mel(y)
    linear = wav_to_linear(y)

    # Save spectrograms
    mel_path = os.path.join(mel_dir, base_id + ".npy")
    linear_path = os.path.join(linear_dir, base_id + ".npy")
    np.save(mel_path, mel)
    np.save(linear_path, linear)

    # Relative paths for metadata
    mel_rel = os.path.join("melspectrograms", base_id + ".npy")
    linear_rel = os.path.join("linear", base_id + ".npy")
    metadata_output.append(f"{base_id}|{linear_rel}|{mel_rel}|{text}")

# Save processed metadata
with open(output_metadata_file, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_output))

print("✅ Done! Tamil metadata and spectrograms saved successfully.")
