"""
Attempt to extract embeddings from emotion2vec checkpoint
Using the pretrained encoder layers to get proper features
"""

import torch
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
import sys

def load_emotion2vec_model():
    """Load emotion2vec checkpoint"""
    model_path = '../emotion2vec_base/emotion2vec_base.pt'
    
    print("Loading emotion2vec checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Get model state dict
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"\nModel state_dict keys: {len(model_state)} layers")
        
        # Print first few keys to understand structure
        print("\nFirst 10 model keys:")
        for i, key in enumerate(list(model_state.keys())[:10]):
            print(f"  {i+1}. {key}: {model_state[key].shape if hasattr(model_state[key], 'shape') else type(model_state[key])}")
    
    return checkpoint, model_state

def extract_encoder_features_direct(audio_path, model_state):
    """
    Try to extract features by manually running audio through encoder layers
    This is a simplified approach without FairSeq
    """
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # [1, T]
    
    # Try to extract features
    # This is a simplified approach - we'd need the full model architecture
    # For now, return None to indicate we need proper integration
    return None

def check_fairseq_availability():
    """Check if FairSeq is available"""
    try:
        import fairseq
        print("✓ FairSeq is available")
        print(f"  Version: {fairseq.__version__}")
        return True
    except ImportError:
        print("✗ FairSeq not available")
        return False

def main():
    print("=" * 70)
    print("EMOTION2VEC ENCODER INTEGRATION CHECK")
    print("=" * 70)
    print()
    
    # Check FairSeq
    has_fairseq = check_fairseq_availability()
    print()
    
    if not has_fairseq:
        print("FairSeq is required to properly use emotion2vec encoder.")
        print("\nOptions:")
        print("  1. Install FairSeq: pip install fairseq")
        print("  2. Use alternative pretrained model (wav2vec2 from transformers)")
        print("  3. Continue with current feature-based approach")
        print()
        
        response = input("Would you like to install FairSeq? (y/n): ")
        if response.lower() == 'y':
            print("\nInstalling FairSeq...")
            os.system("pip install fairseq")
            print("\n✓ Please restart the script after installation")
            return
        else:
            print("\n⚠ Cannot properly integrate emotion2vec without FairSeq")
            print("Consider using wav2vec2 as alternative")
            return
    
    # Load model
    checkpoint, model_state = load_emotion2vec_model()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nTo properly integrate emotion2vec:")
    print("  1. Use FairSeq to load the model")
    print("  2. Extract embeddings from encoder layers")
    print("  3. Train classifier on top of embeddings")
    print("  4. Fine-tune encoder (optional)")

if __name__ == '__main__':
    main()
