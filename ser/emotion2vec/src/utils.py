"""
Utility functions for emotion2vec fine-tuning
"""

import torch
import numpy as np
import soundfile as sf
import os

TARGET_EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']

def load_emotion2vec_model(model_path, device='cpu'):
    """Load emotion2vec pretrained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
        cfg = checkpoint.get('cfg', None)
        return model, cfg
    
    return checkpoint, None

def load_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    waveform, sr = sf.read(audio_path)
    
    # Convert to tensor
    waveform = torch.from_numpy(waveform).float()
    
    # Handle stereo -> mono
    if len(waveform.shape) > 1:
        waveform = waveform.mean(dim=-1)
    
    # Resample if needed
    if sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform

def extract_mfcc_features(waveform, sr=16000, n_mfcc=40):
    """Extract MFCC features as baseline"""
    import librosa
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    
    # Calculate statistics
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    
    # Concatenate features
    features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    
    return features

def get_emotion_index(emotion_name):
    """Get emotion index from name"""
    return TARGET_EMOTIONS.index(emotion_name.lower())

def get_emotion_name(emotion_idx):
    """Get emotion name from index"""
    return TARGET_EMOTIONS[emotion_idx]
