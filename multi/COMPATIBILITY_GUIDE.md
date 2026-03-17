# Transformers API Compatibility Guide

## Quick Fix Reference

### Problem: `evaluation_strategy` not recognized

**Error:**
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

**Solution:**
- **Transformers < 4.30.0**: Use `evaluation_strategy="epoch"`
- **Transformers >= 4.30.0**: Use `eval_strategy="epoch"` ✓ (Current fix)

---

## Common Transformers API Changes

### TrainingArguments Parameters

| Old API (< 4.30) | New API (>= 4.30) | Status |
|------------------|-------------------|--------|
| `evaluation_strategy` | `eval_strategy` | ✓ Fixed |
| `save_strategy` | `save_strategy` | ✓ No change |
| `load_best_model_at_end` | `load_best_model_at_end` | ✓ No change |

### Model Forward Pass

**Issue:** Attention mask dimension mismatch
- **Error:** `RuntimeError: The size of tensor a (218) must match the size of tensor b (69937)`
- **Cause:** Raw audio attention mask doesn't match downsampled hidden states
- **Fix:** Remove manual attention mask handling in pooling (Wav2Vec2 handles it internally)
- **Status:** ✓ Fixed

### Processor/Feature Extractor

**For MMS Models (SER Task)**: Use `Wav2Vec2FeatureExtractor` NOT `Wav2Vec2Processor`

| Task | Use This | Why |
|------|----------|-----|
| **Speech Emotion Recognition** | `Wav2Vec2FeatureExtractor` ✓ | No tokenizer needed |
| **Speech Recognition (ASR)** | `Wav2Vec2Processor` | Needs tokenizer for text |
| **Language ID** | `Wav2Vec2FeatureExtractor` ✓ | Classification only |

**Common Error with MMS:**
```python
# Wrong ❌ - Causes TypeError
processor = Wav2Vec2Processor.from_pretrained("facebook/mms-300m")
# TypeError: expected str, bytes or os.PathLike object, not NoneType

# Correct ✅
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-300m")
```

**Status:** ✓ Fixed in all files

---

## Checking Your Transformers Version

Run this in Kaggle:
```python
import transformers
print(f"Transformers version: {transformers.__version__}")
```

**Expected output:**
- Kaggle (2024+): `4.30.0` or higher
- Local: May vary

---

## If You Still Get Errors

### 1. Check your transformers version
```bash
pip show transformers
```

### 2. Upgrade transformers (if needed)
```bash
pip install --upgrade transformers
```

### 3. Clear Kaggle kernel cache
- Settings → Restart Kernel
- Or: Session → Restart Kernel

### 4. Verify MMS-1B compatibility
```python
from transformers import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained("facebook/mms-1b-all")
print("✓ MMS-1B loaded successfully")
```

---

## Other Potential Issues

### Issue: Attention mask dimension mismatch
**Error:** `RuntimeError: The size of tensor a (218) must match the size of tensor b (69937)`  
**Solution:** Already fixed in the code - uses simple mean pooling without manual attention masking

### Issue: "No space left on device"
**Solution:** MMS-1B is ~4GB. Ensure you have 10GB+ free disk space.

### Issue: "CUDA out of memory"
**Solution:** 
- Reduce `batch_size` from 8 to 4 or 2 ✓ **FIXED**
- Increase `gradient_accumulation_steps` to maintain effective batch size ✓ **FIXED**
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()` ✓ **FIXED**
- Use `fp16=True` (already enabled)
- See `MEMORY_OPTIMIZATION.md` for detailed guide

### Issue: Import errors
**Solution:**
```bash
pip install -q transformers datasets audiomentations scikit-learn librosa soundfile
```

---

## Files Fixed

✓ `multi/mms_multilingual_kaggle.ipynb`
✓ `multi/mms_multilingual_ser.py`

All instances of `evaluation_strategy` replaced with `eval_strategy`.

---

## Quick Test

Run this to verify everything works:
```python
from transformers import TrainingArguments

# This should work without errors
args = TrainingArguments(
    output_dir="./test",
    eval_strategy="epoch",  # New API
    save_strategy="epoch",
    num_train_epochs=1,
)
print("✓ TrainingArguments initialized successfully")
```

---

**Last Updated:** 2026-03-01  
**Transformers Version:** 4.30.0+  
**Status:** ✓ All fixed and tested
