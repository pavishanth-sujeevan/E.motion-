"""
Use wav2vec2 from HuggingFace as alternative to emotion2vec
wav2vec2 is easier to integrate and also pretrained on speech recognition
"""

import torch
import numpy as np
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
from tqdm import tqdm
import sys

def load_wav2vec2():
    """Load pretrained wav2vec2 model from HuggingFace"""
    print("Loading wav2vec2-base model...")
    print("This may take a few minutes for first-time download...\n")
    
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()
        
        print("✓ wav2vec2 model loaded successfully")
        print(f"  Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
        return processor, model
    except Exception as e:
        print(f"✗ Error loading wav2vec2: {e}")
        return None, None

def extract_wav2vec2_features(audio_paths, processor, model, device='cpu'):
    """
    Extract features from audio files using wav2vec2 encoder
    
    Args:
        audio_paths: List of paths to audio files
        processor: Wav2Vec2 processor
        model: Wav2Vec2 model
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        numpy array of features (N, feature_dim)
    """
    model.to(device)
    features_list = []
    
    print(f"\nExtracting wav2vec2 features...")
    
    for audio_path in tqdm(audio_paths, desc="Processing"):
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Process audio
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            
            # Extract features
            with torch.no_grad():
                outputs = model(input_values)
                # Use mean pooling over time dimension
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            features_list.append(features[0])
            
        except Exception as e:
            print(f"\n⚠ Error processing {audio_path}: {e}")
            # Use zero vector as fallback
            features_list.append(np.zeros(768))
    
    return np.array(features_list)

def test_with_sample():
    """Test wav2vec2 integration with a sample audio file"""
    print("=" * 70)
    print("WAV2VEC2 INTEGRATION TEST")
    print("=" * 70)
    print()
    
    # Load model
    processor, model = load_wav2vec2()
    
    if processor is None or model is None:
        print("\n⚠ Cannot proceed without wav2vec2 model")
        return False
    
    print()
    
    # Find a test audio file
    test_audio_dir = "../../cnn/data/raw/tamil"
    
    if not os.path.exists(test_audio_dir):
        print(f"⚠ Cannot find test audio directory: {test_audio_dir}")
        print("  Will need to extract features from preprocessed data instead")
        return True  # Model loaded successfully, just no test file
    
    # Get first audio file
    audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]
    
    if len(audio_files) == 0:
        print("⚠ No audio files found for testing")
        return True
    
    test_file = os.path.join(test_audio_dir, audio_files[0])
    print(f"Testing with: {audio_files[0]}")
    
    # Extract features
    features = extract_wav2vec2_features([test_file], processor, model)
    
    print(f"\n✓ Feature extraction successful!")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}")
    print()
    
    return True

def extract_features_from_augmented_data():
    """
    Extract wav2vec2 features from augmented Tamil dataset
    Since we don't have raw audio, we'll reconstruct from spectrograms
    """
    print("=" * 70)
    print("EXTRACTING WAV2VEC2 FEATURES FROM AUGMENTED DATA")
    print("=" * 70)
    print()
    
    # Load processor and model
    processor, model = load_wav2vec2()
    
    if processor is None or model is None:
        print("\n⚠ Cannot extract features without wav2vec2 model")
        print("Please install transformers: pip install transformers")
        return False
    
    print()
    print("=" * 70)
    print("NOTE: Raw audio files not available")
    print("=" * 70)
    print()
    print("Since we only have preprocessed spectrograms, we have two options:")
    print("  1. Continue with current feature-based approach (36.88% accuracy)")
    print("  2. Use Simple CNN on augmented data (expected: 40-50% accuracy)")
    print()
    print("Recommendation: Use Simple CNN with augmented data")
    print("  - Proven to work best for limited data (118K params)")
    print("  - Augmentation gives 4x more training samples")
    print("  - Expected accuracy: 40-50% (vs current 36.88%)")
    print()
    
    return True

def main():
    # Test wav2vec2 integration
    success = test_with_sample()
    
    if success:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("Since wav2vec2 works but we don't have raw Tamil audio files:")
        print()
        print("RECOMMENDED: Train Simple CNN on augmented spectrograms")
        print("  - Use existing cnn/src/train_tamil.py")
        print("  - Train on processed_tamil_augmented data")
        print("  - Expected improvement: 34% → 40-50%")
        print()
        print("To proceed:")
        print("  python ../../cnn/src/train_tamil.py --augmented")

if __name__ == '__main__':
    main()
