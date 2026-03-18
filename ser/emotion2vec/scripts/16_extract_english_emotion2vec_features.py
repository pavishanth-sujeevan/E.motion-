"""
Extract emotion2vec features from English audio files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
TARGET_SR = 16000
TARGET_LENGTH = 48000  # 3 seconds at 16kHz

# ===== Model Classes =====

class ConvFeatureExtraction(nn.Module):
    def __init__(self, conv_layers_config):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        in_d = 1
        for i, (dim, k, stride) in enumerate(conv_layers_config):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_d, dim, k, stride=stride),
                    nn.Dropout(p=0.0),
                    nn.GroupNorm(dim, dim),
                ))
            in_d = dim
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.gelu(x)
        return x

class SimpleEmotion2VecEncoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.state_dict_full = checkpoint['model']
        
        conv_configs = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 3, 2), (512, 2, 2),
        ]
        
        self.feature_extractor = ConvFeatureExtraction(conv_configs)
        self._load_conv_weights()
        self.post_extract_proj = nn.Linear(512, 768)
    
    def _load_conv_weights(self):
        prefix = "modality_encoders.AUDIO.local_encoder.conv_layers"
        
        for i, conv_block in enumerate(self.feature_extractor.conv_layers):
            conv_weight_key = f"{prefix}.{i}.0.weight"
            if conv_weight_key in self.state_dict_full:
                conv_block[0].weight.data = self.state_dict_full[conv_weight_key]
            
            gn_weight_key = f"{prefix}.{i}.2.1.weight"
            gn_bias_key = f"{prefix}.{i}.2.1.bias"
            if gn_weight_key in self.state_dict_full:
                conv_block[2].weight.data = self.state_dict_full[gn_weight_key]
                conv_block[2].bias.data = self.state_dict_full[gn_bias_key]
    
    def forward(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        features = self.feature_extractor(audio)
        features = features.transpose(1, 2)
        features = self.post_extract_proj(features)
        
        return features

# ===== Audio Processing =====

def load_and_preprocess_audio(file_path):
    """Load audio file and preprocess to fixed length"""
    audio, sr = librosa.load(file_path, sr=TARGET_SR)
    
    # Pad or trim to target length
    if len(audio) < TARGET_LENGTH:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
    else:
        audio = audio[:TARGET_LENGTH]
    
    return audio

def collect_english_files():
    """Collect all English audio files with their labels"""
    data_root = Path('../data/raw')
    files = []
    labels = []
    
    # RAVDESS
    ravdess_path = data_root / 'RAVDESS-SPEECH'
    if ravdess_path.exists():
        for audio_file in ravdess_path.rglob('*.wav'):
            # RAVDESS format: 03-01-05-01-01-01-01.wav
            parts = audio_file.stem.split('-')
            emotion_code = int(parts[2])
            
            emotion_map = {3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 1: 'neutral'}
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                files.append(str(audio_file))
                labels.append(EMOTION_TO_INDEX[emotion])
    
    # TESS
    tess_path = data_root / 'TESS'
    if tess_path.exists():
        for audio_file in tess_path.rglob('*.wav'):
            filename = audio_file.stem.lower()
            
            if 'angry' in filename:
                emotion = 'angry'
            elif 'fear' in filename:
                emotion = 'fear'
            elif 'happy' in filename or 'ps' in filename:
                emotion = 'happy'
            elif 'neutral' in filename:
                emotion = 'neutral'
            elif 'sad' in filename:
                emotion = 'sad'
            else:
                continue
            
            files.append(str(audio_file))
            labels.append(EMOTION_TO_INDEX[emotion])
    
    return files, labels

def extract_features_batch(encoder, file_batch, device):
    """Extract features for a batch of files"""
    audios = []
    for file_path in file_batch:
        audio = load_and_preprocess_audio(file_path)
        audios.append(audio)
    
    # Stack into batch
    audio_batch = torch.FloatTensor(np.stack(audios)).to(device)
    
    # Extract features
    with torch.no_grad():
        features = encoder(audio_batch)  # (batch, time, 768)
        features = features.mean(dim=1)  # (batch, 768) - average pooling
    
    return features.cpu().numpy()

# ===== Main Extraction =====

def main():
    print("=" * 70)
    print("EXTRACTING EMOTION2VEC FEATURES FROM ENGLISH AUDIO")
    print("=" * 70)
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load encoder
    print("Loading emotion2vec encoder...")
    checkpoint_path = '../emotion2vec_base/emotion2vec_base.pt'
    encoder = SimpleEmotion2VecEncoder(checkpoint_path)
    encoder.to(device)
    encoder.eval()
    print("✓ Encoder loaded\n")
    
    # Collect files
    print("Collecting English audio files...")
    files, labels = collect_english_files()
    print(f"✓ Found {len(files)} files\n")
    
    # Class distribution
    print("Class distribution:")
    for emotion, idx in EMOTION_TO_INDEX.items():
        count = labels.count(idx)
        print(f"  {emotion:8s}: {count:4d} samples")
    print()
    
    # Split into train/val/test (70/15/15)
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(files))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    
    labels_temp = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_temp)
    
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}\n")
    
    # Extract features
    batch_size = 16
    
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        print(f"Extracting {split_name} features...")
        
        split_files = [files[i] for i in split_idx]
        split_labels = [labels[i] for i in split_idx]
        
        all_features = []
        
        for i in tqdm(range(0, len(split_files), batch_size), desc=split_name):
            batch_files = split_files[i:i+batch_size]
            batch_features = extract_features_batch(encoder, batch_files, device)
            all_features.append(batch_features)
        
        # Concatenate
        features = np.vstack(all_features)
        labels_array = np.array(split_labels)
        
        # Save
        output_dir = Path('../features/english_emotion2vec')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / f'X_{split_name}.npy', features)
        np.save(output_dir / f'y_{split_name}.npy', labels_array)
        
        print(f"✓ Saved {split_name}: {features.shape}\n")
    
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nFeatures saved to: {output_dir}")
    print(f"Shape: (samples, 768)")

if __name__ == '__main__':
    main()
