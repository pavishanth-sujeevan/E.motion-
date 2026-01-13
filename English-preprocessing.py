import re
import matplotlib.pyplot as plt
import pandas as pd
import unidecode
from num2words import num2words
import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.signal import stft

#ENGLISH TEXT PREPROCESSING
#load dataset
metadata_file="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech_Model\\data\\raw\\english\\metadata.tsv"
audio_folder="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech_Model\\data\\raw\\english\\wavs"

df=pd.read_csv(metadata_file,sep="|",names=["audio_path","text"])
df['audio_path']=df['audio_path'].apply(lambda x: os.path.join(audio_folder,os.path.basename(x.strip())))

#Basic info
print("Dataset shape: ", df.shape)
print("\nColumns: ", df.columns)
print("\nFirst 5 rows: \n", df.head())

#Check for missing values
print(df.isnull().sum())

#Check for duplicates
print("\nNumber of duplicate rows: ",df.duplicated().sum())

#Remove rows with missing or empty text/audio
df=df.dropna(subset=["audio_path","text"])
df=df[df["text"].str.strip().astype(bool)]
df=df.drop_duplicates(subset=["audio_path","text"])

# abbreviations dictionary
abbreviations = {
    "mr.": "mister",
    "mrs.": "misses",
    "ms.": "miss",
    "dr.": "doctor",
    "prof.": "professor",
    "e.g": "for example",
    "i.e": "that is",
    "etc.": "and so on",
    "vs.": "versus",
    "st.": "street",
    "ave.": "avenue",
    "rd.": "road",
    "mt.": "mount",
    "lt.": "lieutenant",
    "capt.": "captain",
    "col.": "colonel",
    "gen.": "general",
    "sgt.": "sergeant",
    "sr.": "senior",
    "jr.": "junior",
    "a.m.": "morning",
    "p.m.": "evening",
    "inc.": "incorporated",
    "ltd.": "limited",
    "co.": "company",
    "corp.": "corporation"
}


def expand_abbreviations(text):
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    return text

def convert_numbers_in_text(text):
    def convert_number(match):
        try:
            return num2words(int(match.group()))
        except:
            return match.group()
    return re.sub(r'\d+', convert_number, text)

def clean_text(text):
    text=unidecode.unidecode(text)  #normalize unicode characters
    text = text.strip().lower()
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s-]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = expand_abbreviations(text)
    text = convert_numbers_in_text(text)
    return text 

# load and clean CSV
def load_and_clean(path, cleaner):
    df = pd.read_csv(path, sep="|", names=["audio_path", "text"])
    df = df.dropna(subset=["audio_path", "text"])
    df = df[df["text"].str.strip().astype(bool)].drop_duplicates(["audio_path", "text"])
    df["cleaned_text"] = df["text"].apply(cleaner)
    return df



df_cleaned = pd.DataFrame({
    "audio_path": df['audio_path'].apply(lambda x: os.path.basename(x)), 
    "text": [clean_text(t) for t in df['text']]  # apply cleaning to all text
})

# Save to TSV 
output_file = "C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\english\\eng_metadata_cleaned.tsv"
df_cleaned.to_csv(output_file, sep="|", index=False, header=False)

print("\nCleaned metadata with audio path and text saved successfully!")


#ENGLISH AUDIO PREPROCESSING
# CONFIGURATION
wav_dir="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\raw\\english\\wavs"
metadata_file="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\english\\eng_metadata_cleaned.tsv"

mel_dir="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\english\\melspectrograms"
linear_dir="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\english\\linear"

output_metadata_file="C:\\Users\\amawg\\OneDrive\\Documents\\IIT 2nd Year\\DSGP\\Text-to-Speech-Model\\data\\processed\\english\\eng_ metadata_processed.tsv"

os.makedirs(mel_dir,exist_ok=True)
os.makedirs(linear_dir,exist_ok=True)

sample_rate=22050
n_fft=1024
hop_length=256
win_length=1024
n_mels=80
fmin=0
fmax=8000

#mel function
def wav_to_mel(y):
    mel=librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_db=librosa.power_to_db(mel,ref=np.max)
    return mel_db

#linear spectrogram
def wav_to_linear(y):
    D=librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    magnitude=np.abs(D)
    magnitude_db=librosa.amplitude_to_db(magnitude,ref=np.max)
    return magnitude_db

#processing
metadata_output=[]

with open(metadata_file,"r",encoding="utf-8")as f:
    lines=f.readlines()

print("Processing audio....")

for line in tqdm(lines):
    wav_id,text=line.strip().split("|")

    wav_filename=wav_id
    if wav_filename.endswith(".wav"):
        wav_filename=wav_filename[:-4]

    wav_path=os.path.join(wav_dir,wav_filename+".wav")    

    y,sr=librosa.load(wav_path,sr=sample_rate)

    #remove silence
    y,_=librosa.effects.trim(y,top_db=20)

    #normalize
    y=y/(np.max(np.abs(y))+1e-9)

    #generate spectrograms
    mel=wav_to_mel(y)
    linear=wav_to_linear(y)

    #save as numpy files
    mel_path=os.path.join(mel_dir,wav_id+".npy")
    linear_path=os.path.join(linear_dir,wav_id+".npy")

    np.save(mel_path,mel)
    np.save(linear_path,linear)

    #add to processed metadata
    mel_rel=os.path.join("melspectrograms",wav_id+".npy")
    linear_rel=os.path.join("linear",wav_id+".npy")
    
    metadata_output.append(f"{wav_id}|{linear_rel}|{mel_rel}|{text}")

    #save metadata file
    with open(output_metadata_file,"w",encoding="utf-8")as f:
        f.write("\n".join(metadata_output))

    print("Done!")
    

