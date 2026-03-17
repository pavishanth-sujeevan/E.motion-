"""
Attempt to load emotion2vec checkpoint without FairSeq
Extract the model weights and build a compatible encoder
"""

import torch
import torch.nn as nn
import numpy as np
import os

def inspect_checkpoint():
    """Thoroughly inspect the emotion2vec checkpoint structure"""
    print("=" * 70)
    print("EMOTION2VEC CHECKPOINT INSPECTION")
    print("=" * 70)
    print()
    
    checkpoint_path = '../emotion2vec_base/emotion2vec_base.pt'
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nTop-level keys: {list(checkpoint.keys())}")
    print()
    
    # Inspect 'args' if available
    if 'args' in checkpoint:
        args = checkpoint['args']
        print("Arguments:")
        if hasattr(args, '__dict__'):
            for key, value in vars(args).items():
                print(f"  {key}: {value}")
        print()
    
    # Inspect 'cfg' if available
    if 'cfg' in checkpoint:
        print("Configuration:")
        cfg = checkpoint['cfg']
        print(f"  Type: {type(cfg)}")
        if hasattr(cfg, '__dict__'):
            for key in dir(cfg):
                if not key.startswith('_'):
                    try:
                        value = getattr(cfg, key)
                        if not callable(value):
                            print(f"  {key}: {value}")
                    except:
                        pass
        print()
    
    # Inspect model state dict
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"Model state_dict: {len(model_state)} keys")
        print("\nFirst 20 layer names and shapes:")
        for i, (key, value) in enumerate(list(model_state.items())[:20]):
            if hasattr(value, 'shape'):
                print(f"  {i+1}. {key}: {value.shape}")
            else:
                print(f"  {i+1}. {key}: {type(value)}")
        
        print("\nLooking for encoder patterns...")
        encoder_keys = [k for k in model_state.keys() if 'encoder' in k.lower()]
        print(f"Found {len(encoder_keys)} encoder-related keys")
        
        if encoder_keys:
            print("\nEncoder layers:")
            for key in encoder_keys[:10]:
                if hasattr(model_state[key], 'shape'):
                    print(f"  {key}: {model_state[key].shape}")
    
    return checkpoint

def try_extract_features_manually(checkpoint):
    """
    Try to extract features by manually applying transformations
    This is a research approach to understand the model structure
    """
    print("\n" + "=" * 70)
    print("ATTEMPTING MANUAL FEATURE EXTRACTION")
    print("=" * 70)
    print()
    
    if 'model' not in checkpoint:
        print("❌ No model state found in checkpoint")
        return None
    
    model_state = checkpoint['model']
    
    # Look for key layers
    print("Searching for critical layers:")
    
    # Common patterns in wav2vec2-style models
    patterns = [
        'feature_extractor',
        'post_extract_proj',
        'encoder.layers',
        'mask_emb',
        'layer_norm',
        'project_q',
        'final_proj'
    ]
    
    for pattern in patterns:
        matching = [k for k in model_state.keys() if pattern in k]
        if matching:
            print(f"  ✓ Found {pattern}: {len(matching)} layers")
        else:
            print(f"  ✗ Not found: {pattern}")
    
    print()
    print("This model appears to be a FairSeq checkpoint.")
    print("We need the model architecture definition to properly load it.")
    
    return model_state

def main():
    try:
        checkpoint = inspect_checkpoint()
        
        if checkpoint:
            model_state = try_extract_features_manually(checkpoint)
            
            print("\n" + "=" * 70)
            print("CONCLUSION")
            print("=" * 70)
            print()
            print("The checkpoint structure is clear, but we need:")
            print("  1. The model architecture definition, OR")
            print("  2. FairSeq library to load it properly, OR")
            print("  3. emotion2vec-plus (newer, HuggingFace-compatible version)")
            print()
            print("Next steps:")
            print("  - Try installing FairSeq with older pip")
            print("  - Look for emotion2vec_plus or emotion2vec_base_finetuned")
            print("  - Try Python 3.10 environment")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
