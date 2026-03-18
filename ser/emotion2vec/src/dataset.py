import torch
import pandas as pd
import librosa
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, metadata_path, split='train'):
        df = pd.read_csv(metadata_path)
        self.df = df[df['split'] == split].reset_index(drop=True)

        # Hugging Face emotion2vec_plus_base 9-class mapping:
        # 0: angry, 1: disgusted, 2: fearful, 3: happy,
        # 4: neutral, 5: other, 6: sad, 7: surprised, 8: unknown
        self.label_to_idx = {
            'angry': 0,
            'disgust': 1,
            'fearful': 2,
            'happy': 3,
            'neutral': 4,
            'calm': 4,  # Map 'calm' to 'neutral'
            'sad': 6,
            'surprised': 7,
            'surprise': 7  # Standardize TESS/RAVDESS naming
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load audio at 16kHz (already preprocessed)
        waveform, _ = librosa.load(row['path'], sr=16000)

        # Get numerical label from our map
        label_str = row['label']
        label = self.label_to_idx.get(label_str, 8)  # Default to 'unknown' (8) if not found

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)