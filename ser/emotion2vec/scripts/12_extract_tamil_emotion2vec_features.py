"""
Extract emotion2vec features from Tamil (EmoTa) dataset
Use our custom encoder to get proper embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Emotion mappings
EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

# ===== Custom Encoder (copied from working implementation) =====

class ConvFeatureExtraction(nn.Module):
    """Conv feature extraction module from emotion2vec"""
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
    """Simplified emotion2vec encoder for feature extraction"""
    def __init__(self, checkpoint_path):
        super().__init__()
        
        print("Loading emotion2vec encoder...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.state_dict_full = checkpoint['model']
        
        conv_configs = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 3, 2), (512, 2, 2),
        ]
        
        self.feature_extractor = ConvFeatureExtraction(conv_configs)
        self._load_conv_weights()
        self.post_extract_proj = nn.Linear(512, 768)
        
        print("✓ Encoder loaded")
    
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

# ===== Feature Extraction =====

def load_tamil_audio_files(data_dir):
    """Load Tamil audio files from EmoTa dataset"""
    audio_files = []
    labels = []
    
    print(f"Loading Tamil audio from: {data_dir}")
    
    # EmoTa naming: XX_XX_emotion.wav
    emotion_map = {
        'ang': 'angry',
        'fea': 'fear',
        'hap': 'happy',
        'neu': 'neutral',
        'sad': 'sad'
    }
    
    for wav_file in glob.glob(os.path.join(data_dir, '*.wav')):
        filename = os.path.basename(wav_file)
        
        # Parse emotion from filename
        parts = filename.replace('.wav', '').split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                audio_files.append(wav_file)
                labels.append(EMOTION_TO_INDEX[emotion])
    
    print(f"  Found {len(audio_files)} audio files")
    
    # Count per emotion
    print("\nPer-emotion counts:")
    for emotion_idx in range(5):
        count = labels.count(emotion_idx)
        print(f"  {INDEX_TO_EMOTION[emotion_idx]:8s}: {count}")
    
    return audio_files, labels

def extract_features_batch(encoder, audio_files, device='cpu', batch_size=16):
    """Extract features with batching for speed"""
    encoder.to(device)
    encoder.eval()
    
    features_list = []
    
    print("\nExtracting emotion2vec features...")
    
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Batches"):
        batch_files = audio_files[i:i+batch_size]
        batch_audios = []
        
        for audio_file in batch_files:
            try:
                # Load audio
                audio, sr = sf.read(audio_file)
                
                # Convert to mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Pad or trim to 3 seconds (48000 samples)
                target_length = 48000
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                
                batch_audios.append(audio)
                
            except Exception as e:
                print(f"\n⚠ Error loading {audio_file}: {e}")
                batch_audios.append(np.zeros(48000))
        
        # Convert to batch tensor
        batch_tensor = torch.FloatTensor(np.array(batch_audios)).to(device)
        
        # Extract features
        with torch.no_grad():
            features = encoder(batch_tensor)  # (batch, time, 768)
            features_pooled = features.mean(dim=1)  # (batch, 768)
        
        features_list.extend(features_pooled.cpu().numpy())
    
    return np.array(features_list)

def main():
    print("=" * 70)
    print("EMOTION2VEC FEATURE EXTRACTION - TAMIL DATASET")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load encoder
    checkpoint_path = '../emotion2vec_base/emotion2vec_base.pt'
    encoder = SimpleEmotion2VecEncoder(checkpoint_path)
    print()
    
    # Load Tamil audio files
    data_dir = '../../cnn/data/raw/EmoTa/TamilSER-DB'
    audio_files, labels = load_tamil_audio_files(data_dir)
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return
    
    # Split data (70/15/15)
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        test_files, test_labels, test_size=0.5, stratify=test_labels, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")
    
    # Extract features
    X_train = extract_features_batch(encoder, train_files, device, batch_size=16)
    X_val = extract_features_batch(encoder, val_files, device, batch_size=16)
    X_test = extract_features_batch(encoder, test_files, device, batch_size=16)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Save features
    output_dir = '../features/tamil_emotion2vec'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), train_labels)
    np.save(os.path.join(output_dir, 'y_val.npy'), val_labels)
    np.save(os.path.join(output_dir, 'y_test.npy'), test_labels)
    
    print(f"\n✓ Features saved to: {output_dir}")
    print()
    print("Next step: Train classifier on these features!")
    print("  Expected accuracy: 50-65% (vs 36.88% baseline)")

if __name__ == '__main__':
    main()
