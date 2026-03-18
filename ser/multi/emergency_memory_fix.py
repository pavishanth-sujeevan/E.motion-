# EMERGENCY MEMORY FIX - Run this cell BEFORE training

import os
import gc
import torch

print("=" * 80)
print("APPLYING EMERGENCY MEMORY OPTIMIZATIONS")
print("=" * 80)

# 1. Set PyTorch memory allocator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("✓ Set expandable_segments:True")

# 2. Clear ALL GPU cache
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("✓ GPU cache cleared")

# 3. Update config for minimal memory
config.phase_a_batch_size = 1  # Absolute minimum
config.phase_b_batch_size = 1
config.gradient_accumulation_steps = 32  # Maintain effective batch=32
config.max_duration = 6.0  # Shorter audio
config.freeze_layers = 35  # Freeze more layers

print("\nUpdated Configuration:")
print(f"  Batch size: {config.phase_a_batch_size} (minimum)")
print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
print(f"  Effective batch: {config.phase_a_batch_size * config.gradient_accumulation_steps}")
print(f"  Max audio duration: {config.max_duration}s (reduced)")
print(f"  Frozen layers: {config.freeze_layers}/48")

# 4. Check current GPU usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU Memory Status:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Total: {total:.2f} GB")
    print(f"  Free: {total - allocated:.2f} GB")

# 5. Enable gradient checkpointing if not already
try:
    model.gradient_checkpointing_enable()
    print("\n✓ Gradient checkpointing enabled")
except:
    print("\n⚠ Note: Enable gradient checkpointing after model initialization")

print("\n" + "=" * 80)
print("READY TO TRAIN WITH MINIMAL MEMORY CONFIGURATION")
print("=" * 80)
print("\nIf this still fails, consider:")
print("1. Using MMS-300M instead of MMS-1B")
print("2. Reducing max_duration to 4.0 seconds")
print("3. Training on a larger GPU (A100, V100)")
