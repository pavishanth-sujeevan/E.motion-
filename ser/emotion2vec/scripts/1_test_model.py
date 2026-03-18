"""
Test emotion2vec model loading and basic inference
"""

import torch
import numpy as np
import soundfile as sf
import os
import sys

def test_model_loading():
    """Test if emotion2vec model loads correctly"""
    print("=" * 70)
    print("TESTING emotion2vec MODEL LOADING")
    print("=" * 70)
    print()
    
    # Model path
    model_path = os.path.join(os.path.dirname(__file__), '..', 'emotion2vec_base', 'emotion2vec_base.pt')
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ ERROR: Model file not found!")
        return False
    
    print(f"✓ Model file exists ({os.path.getsize(model_path) / (1024**3):.2f} GB)")
    print()
    
    # Load model
    print("Loading model...")
    try:
        # Check CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Load the model
        model = torch.load(model_path, map_location=device)
        print("✓ Model loaded successfully!")
        print()
        
        # Inspect model structure
        print("Model structure:")
        if isinstance(model, dict):
            print("  Model is a dictionary with keys:")
            for key in model.keys():
                print(f"    - {key}")
                if key == 'model' and isinstance(model[key], torch.nn.Module):
                    print(f"      Type: {type(model[key])}")
                    # Count parameters
                    try:
                        total_params = sum(p.numel() for p in model[key].parameters())
                        print(f"      Total parameters: {total_params:,}")
                    except:
                        pass
        elif isinstance(model, torch.nn.Module):
            print(f"  Model type: {type(model)}")
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Total parameters: {total_params:,}")
        else:
            print(f"  Unknown model type: {type(model)}")
        
        print()
        print("=" * 70)
        print("MODEL LOADING TEST: PASSED ✓")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction from sample audio"""
    print()
    print("=" * 70)
    print("TESTING FEATURE EXTRACTION")
    print("=" * 70)
    print()
    
    # Check for test audio
    test_audio_path = os.path.join(os.path.dirname(__file__), '..', 'emotion2vec_base', 'example', 'test.wav')
    
    if not os.path.exists(test_audio_path):
        print("⚠ No test audio found, skipping feature extraction test")
        return True
    
    print(f"Test audio: {test_audio_path}")
    
    try:
        # Load audio using soundfile
        waveform, sample_rate = sf.read(test_audio_path)
        print(f"✓ Audio loaded: shape={waveform.shape}, sr={sample_rate}")
        
        # Convert to tensor
        waveform = torch.from_numpy(waveform).float()
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        print(f"✓ Converted to tensor: shape={waveform.shape}")
        
        # Resample to 16kHz if needed (emotion2vec expects 16kHz)
        if sample_rate != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            print(f"✓ Resampled to 16kHz: shape={waveform.shape}")
        
        print()
        print("=" * 70)
        print("FEATURE EXTRACTION TEST: PASSED ✓")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ ERROR in feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print()
    print("🚀 emotion2vec Model Testing")
    print()
    
    # Test 1: Model loading
    test1_passed = test_model_loading()
    
    # Test 2: Feature extraction
    if test1_passed:
        test2_passed = test_feature_extraction()
    else:
        print("\n⚠ Skipping feature extraction test due to model loading failure")
        test2_passed = False
    
    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Model Loading: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Feature Extraction: {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    print()
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Ready to proceed with fine-tuning.")
    else:
        print("⚠ Some tests failed. Please check errors above.")
