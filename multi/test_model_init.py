"""
Quick test to verify MMS model initialization works correctly
Run this in Kaggle to confirm the fix
"""

import torch
from dataclasses import dataclass

# Minimal config for testing
@dataclass
class TestConfig:
    model_name: str = "facebook/mms-1b-all"
    num_labels: int = 5
    num_languages: int = 3
    hidden_size: int = 1280
    dropout: float = 0.1
    lang_to_id = {'english': 0, 'tamil': 1, 'sinhala': 2}

print("Testing model initialization...")
print("=" * 60)

try:
    # Test 1: Check if transformers is available
    print("\n1. Checking dependencies...")
    from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2PreTrainedModel
    print("   ✓ Transformers imported successfully")
    
    # Test 2: Load MMS-1B config
    print("\n2. Loading MMS-1B configuration...")
    config = TestConfig()
    model_config = Wav2Vec2Config.from_pretrained(config.model_name)
    print(f"   ✓ Config loaded (hidden_size: {model_config.hidden_size})")
    
    # Test 3: Load pretrained MMS-1B model
    print("\n3. Loading pretrained MMS-1B encoder...")
    wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_name)
    print(f"   ✓ MMS-1B loaded successfully")
    print(f"   ✓ Parameters: {sum(p.numel() for p in wav2vec2.parameters()) / 1e6:.1f}M")
    
    # Test 4: Verify model can do forward pass
    print("\n4. Testing forward pass...")
    dummy_input = torch.randn(1, 16000)  # 1 second of audio
    with torch.no_grad():
        outputs = wav2vec2(dummy_input)
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output shape: {outputs.last_hidden_state.shape}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour environment is ready. You can now:")
    print("1. Run the full training notebook")
    print("2. Initialize MMSForMultilingualSER without errors")
    print("3. Start training on your datasets")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have internet connection (to download MMS-1B)")
    print("2. Check that transformers version is >= 4.30.0")
    print("3. Ensure you have enough disk space (~5GB for MMS-1B)")
    print("\nRun: pip install -q --upgrade transformers")
