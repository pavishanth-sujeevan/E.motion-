import re
import pandas as pd
import os
import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.signal import stft


#Sinhala character set
vowels = ['අ','ආ','ඇ','ඈ','ඉ','ඊ','උ','ඌ','ඍ','ඎ','එ','ඒ','ඓ','ඔ','ඕ','ඖ']

vowel_strokes = ['්','ා','ැ','ෑ','ි','ී','ු','ූ','ෘ','ෲ','ෙ','ේ','ෛ','ො','ෝ','ෞ','ෟ','ෳ']

consonants = ['ක','ඛ','ග','ඝ','ඞ','ඟ','ච','ඡ','ජ','ඣ','ඤ','ඥ','ඦ','ට','ඨ','ඩ','ඪ','ණ',
              'ත','ථ','ද','ධ','න','ප','ඵ','බ','භ','ම','ය','ර','ල','ව','ශ','ෂ','ස','හ','ළ','ෆ','ට','ඨ']

sinhala_chars = set(vowels + vowel_strokes + consonants)

#load dataset
metadata_file="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\raw\\Sinhala\\metadata.tsv"
audio_folder="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\raw\\Sinhala\\wavs"


df=pd.read_csv(metadata_file,sep="\t",names=["sentence","file_path"],encoding="utf-8", header=0)
df["file_path"]=df['file_path'].fillna('')
df['file_path']=df['file_path'].apply(lambda x: os.path.join(audio_folder,os.path.basename(x.strip())))

#Basic info
print("Dataset shape: ", df.shape)
print("\nColumns: ", df.columns)
print("\nFirst 5 rows: \n", df.head())

#Check for missing values
print("Number of missing values")
print(df.isnull().sum())

#Check for duplicates
print("\nNumber of duplicate rows: ",df.duplicated().sum())

#Remove rows with missing or empty text/audio
df=df.dropna(subset=["sentence","file_path"])
df=df[df["sentence"].str.strip().astype(bool)]
df=df.drop_duplicates(subset=["sentence","file_path"])

def normalize_currency(text):
    text = re.sub(r"\bRs\.?\b", "රුපියල්", text)  
    text = re.sub(r"\$", "ඩොලර්", text)
    text = re.sub(r"£", "පවුම්", text)
    return text

abbreviations={
    "ප.ව." :  "පස්වරු",
   " ෙප.ව." : "පෙරවරු"
}

def normalize_abbreviations(text):
    for abbr, pron in abbreviations.items():
        text = re.sub(re.escape(abbr), pron, text)
    return text


#clean text
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    #text = re.sub(r"[^අ-ෆා-ෞ0-9\s\.,\?!']", "", text)
    #text = re.sub(r"[a-zA-Z]", "", text)
    #text = re.sub(r"\d+", lambda m: convert_numbers_to_sinhala(int(m.group())), text)
    #text = re.sub(r"[^\u0D80-\u0DFF .,?!']", "", text)
    text = normalize_currency(text)
    text = normalize_abbreviations(text)

    return text

# Apply cleaning
df['cleaned_text'] = df['sentence'].apply(clean_text)


# Make sure cleaned_texts matches df length
df_cleaned = pd.DataFrame({
    "audio_path": df['file_path'].apply(lambda x:os.path.split(x.strip())[-1]), 
    "text": df["cleaned_text"]  # apply cleaning to all text
})

df_cleaned = df_cleaned[['audio_path', 'text']]

# Save to TSV 
output_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\sinhala\\sin_metadata_cleaned.tsv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_cleaned.to_csv(output_file, sep="|", index=False, header=False)

print("\nCleaned metadata with audio path and text saved successfully!")



#SINHALA AUDIO PREPROCESSING
#CONFIGURATION
wav_dir = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\raw\\Sinhala\\wavs"
metadata_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\sinhala\\sin_metadata_cleaned.tsv"

mel_dir = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\sinhala\\melspectrograms"
linear_dir = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\sinhala\\linear"
output_metadata_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\sinhala\\sin_metadata_processed.tsv"

os.makedirs(mel_dir, exist_ok=True)
os.makedirs(linear_dir, exist_ok=True)

sample_rate = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 8000

# mel spectrograms
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

# linear spectrograms
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

# processing
metadata_output = []

with open(metadata_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Processing Sinhala audio...")

for line in tqdm(lines):
    wav_id, text = line.strip().split("|")
    base_id=os.path.splitext(wav_id)[0]

    wav_path = os.path.join(wav_dir, wav_id)
    if not wav_path.endswith(".wav"):
        wav_path += ".wav"

    # Load audio
    y, sr = librosa.load(wav_path, sr=sample_rate)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)

    # Generate spectrograms
    mel = wav_to_mel(y)
    linear = wav_to_linear(y)

    # Save as numpy files
    mel_path = os.path.join(mel_dir, base_id + ".npy")
    linear_path = os.path.join(linear_dir, base_id + ".npy")

    np.save(mel_path, mel)
    np.save(linear_path, linear)

    # relative paths
    mel_rel = os.path.join("melspectrograms", base_id + ".npy")
    linear_rel = os.path.join("linear", base_id + ".npy")

    metadata_output.append(f"{base_id}|{linear_rel}|{mel_rel}|{text}")

# Save metadata file
with open(output_metadata_file, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_output))

print("✅ Done! Sinhala metadata and spectrograms saved successfully.")
