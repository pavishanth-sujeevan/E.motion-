"""
Try emotion2vec_plus_base - newer version that should work with transformers
"""

import torch
import numpy as np
from transformers import AutoModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import soundfile as sf
import os
from tqdm import tqdm

def try_emotion2vec_plus():
    """Try loading emotion2vec_plus which might be transformers-compatible"""
    print("=" * 70)
    print("TRYING EMOTION2VEC_PLUS_BASE")
    print("=" * 70)
    print()
    
    model_name = "emotion2vec/emotion2vec_plus_base"
    
    print(f"Loading model: {model_name}")
    print("This may take a few minutes for first download...")
    print()
    
    try:
        # Try loading with AutoModel
        print("Attempting with AutoModel...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Type: {type(model)}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Try to get feature extractor
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            print(f"  ✓ Feature extractor loaded")
        except:
            print(f"  ⚠ No feature extractor, will use manual preprocessing")
            feature_extractor = None
        
        return model, feature_extractor
        
    except Exception as e:
        print(f"❌ Failed with AutoModel: {e}")
        print()
        
        # Try alternative: use model from_pretrained with trust_remote_code
        try:
            print("Attempting alternative loading method...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print(f"  Config: {config}")
            
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            print("✓ Model loaded with alternative method")
            return model, None
            
        except Exception as e2:
            print(f"❌ Also failed: {e2}")
            return None, None

def test_feature_extraction(model, feature_extractor):
    """Test extracting features from a sample audio"""
    print("\n" + "=" * 70)
    print("TESTING FEATURE EXTRACTION")
    print("=" * 70)
    print()
    
    # Find a test audio file
    test_dirs = [
        "../../cnn/data/raw/tamil",
        "../../cnn/data/raw/english",
    ]
    
    test_file = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
            if files:
                test_file = os.path.join(test_dir, files[0])
                break
    
    if not test_file:
        print("⚠ No test audio file found")
        return False
    
    print(f"Test file: {test_file}")
    
    try:
        # Load audio
        audio, sr = sf.read(test_file)
        
        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        print(f"  Audio loaded: {audio.shape}, {sr}Hz")
        
        # Extract features
        model.eval()
        
        if feature_extractor:
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values
        else:
            # Manual preprocessing
            input_values = torch.FloatTensor(audio).unsqueeze(0)
        
        print(f"  Input shape: {input_values.shape}")
        
        with torch.no_grad():
            outputs = model(input_values)
            
            # Try to get embeddings
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                print(f"  ✓ Embeddings shape: {embeddings.shape}")
                print(f"  ✓ Feature dimension: {embeddings.shape[-1]}")
                return True
            elif hasattr(outputs, 'hidden_states'):
                embeddings = outputs.hidden_states[-1]
                print(f"  ✓ Embeddings shape: {embeddings.shape}")
                return True
            else:
                print(f"  Output type: {type(outputs)}")
                print(f"  Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                return False
                
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    model, feature_extractor = try_emotion2vec_plus()
    
    if model is not None:
        success = test_feature_extraction(model, feature_extractor)
        
        if success:
            print("\n" + "=" * 70)
            print("SUCCESS!")
            print("=" * 70)
            print()
            print("✓ emotion2vec_plus_base is working!")
            print("✓ Can extract features from audio")
            print()
            print("Next steps:")
            print("  1. Extract features from Tamil dataset")
            print("  2. Train classification head")
            print("  3. Fine-tune encoder layers")
        else:
            print("\n⚠ Model loaded but feature extraction needs debugging")
    else:
        print("\n" + "=" * 70)
        print("FALLBACK OPTIONS")
        print("=" * 70)
        print()
        print("emotion2vec_plus didn't work either. Remaining options:")
        print("  1. Try thegenerativegeneration/emotion2vec_base_finetuned")
        print("  2. Install FairSeq on Linux/WSL")
        print("  3. Build custom encoder from checkpoint weights")

if __name__ == '__main__':
    main()
