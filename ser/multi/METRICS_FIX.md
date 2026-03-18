# ✅ Metrics Computation Error Fixed

## The Problem

You got this error during evaluation:
```
InvalidParameterError: The 'y_pred' parameter of accuracy_score must be an array-like 
or a sparse matrix. Got np.int64(12) instead.
```

## Root Cause

**Two issues:**

### Issue 1: Missing 'logits' Key in Model Output

The HuggingFace `Trainer` expects models to return predictions in a specific format:
- For training: Return a dictionary with `'loss'` key
- For evaluation: Return `'logits'` key (the predictions before argmax)

Our model was returning:
```python
return {
    'loss': total_loss,
    'emotion_logits': emotion_logits,  # ❌ Trainer doesn't know to use this
    'lang_logits': lang_logits
}
```

**The Trainer couldn't find 'logits', so it used the wrong values for predictions.**

### Issue 2: Incorrect Prediction Indexing

The `compute_metrics` function was incorrectly accessing predictions:
```python
preds = pred.predictions[0].argmax(-1)  # ❌ This gets a scalar, not array
```

When `pred.predictions[0]` was applied to the wrong shape, it indexed a single element instead of the batch.

## The Fix

### Fix 1: Add 'logits' Key to Model Output

```python
# In forward() method
return {
    'loss': total_loss,
    'logits': (emotion_logits, lang_logits),  # ✅ Trainer uses this for predictions
    'emotion_logits': emotion_logits,
    'lang_logits': lang_logits,
    'lang_probs': lang_probs
}
```

Now the Trainer knows what to use for predictions!

### Fix 2: Properly Handle Predictions in compute_metrics

```python
def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    
    # predictions is a tuple: (emotion_logits, lang_logits)
    # We only need emotion_logits for metrics
    if isinstance(pred.predictions, tuple):
        emotion_logits = pred.predictions[0]  # ✅ First element is emotion_logits
    else:
        emotion_logits = pred.predictions
    
    preds = emotion_logits.argmax(-1)  # ✅ Now this is an array
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }
```

## How It Works

### Trainer's Prediction Flow

1. **During Forward Pass:**
   ```python
   outputs = model(input_values=..., labels=...)
   # outputs = {'loss': ..., 'logits': (emotion_logits, lang_logits), ...}
   ```

2. **Trainer Extracts Logits:**
   ```python
   logits = outputs['logits']  # Gets (emotion_logits, lang_logits)
   ```

3. **Accumulates Predictions:**
   ```python
   # Trainer collects all logits across batches
   all_predictions = [batch1_logits, batch2_logits, ...]
   # Converts to tuple if multiple outputs
   ```

4. **Calls compute_metrics:**
   ```python
   compute_metrics(
       EvalPrediction(
           predictions=(all_emotion_logits, all_lang_logits),  # Tuple!
           label_ids=all_labels
       )
   )
   ```

5. **We Extract Emotion Logits:**
   ```python
   emotion_logits = pred.predictions[0]  # Shape: [num_samples, num_emotions]
   preds = emotion_logits.argmax(-1)      # Shape: [num_samples]
   ```

### Why This Matters

**Before Fix:**
- Trainer didn't find 'logits' key
- Used wrong values → shape mismatch
- `predictions[0]` on wrong shape → scalar instead of array
- `accuracy_score` got scalar → error

**After Fix:**
- Trainer finds 'logits' key
- Uses correct emotion logits → proper shape
- `predictions[0]` gets emotion logits array → correct
- `accuracy_score` gets array → works!

## What Changed

### Files Updated
- ✅ `mms_multilingual_kaggle.ipynb` - Both forward() and compute_metrics()
- ✅ `mms_multilingual_ser.py` - Both forward() and compute_metrics()

### Model Forward Method
```python
# Added this line to return dictionary:
'logits': (emotion_logits, lang_logits),  # Trainer expects 'logits' key
```

### Compute Metrics Function
```python
# Changed from:
preds = pred.predictions[0].argmax(-1)  # ❌ Wrong

# To:
if isinstance(pred.predictions, tuple):
    emotion_logits = pred.predictions[0]
else:
    emotion_logits = pred.predictions
preds = emotion_logits.argmax(-1)  # ✅ Correct
```

## Verification

To verify the fix works, you can add debug logging:

```python
def compute_metrics(pred):
    labels = pred.label_ids
    
    # Debug: Check shapes
    print(f"predictions type: {type(pred.predictions)}")
    if isinstance(pred.predictions, tuple):
        print(f"predictions[0] shape: {pred.predictions[0].shape}")
        emotion_logits = pred.predictions[0]
    else:
        print(f"predictions shape: {pred.predictions.shape}")
        emotion_logits = pred.predictions
    
    preds = emotion_logits.argmax(-1)
    print(f"labels shape: {labels.shape}")
    print(f"preds shape: {preds.shape}")
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }
```

**Expected output during evaluation:**
```
predictions type: <class 'tuple'>
predictions[0] shape: (num_eval_samples, 5)  # 5 emotions
labels shape: (num_eval_samples,)
preds shape: (num_eval_samples,)
```

## Common Pitfall: Multi-Output Models

This issue is common when working with models that output multiple things:
- Emotion logits
- Language logits
- Other auxiliary outputs

**Rule of thumb for Trainer:**
- **Always return 'logits' key** with what you want to evaluate
- If multiple outputs, return tuple: `'logits': (output1, output2)`
- In `compute_metrics`, extract the one you need: `pred.predictions[0]`

## Summary

✅ **Added 'logits' key to model output**  
✅ **Fixed prediction indexing in compute_metrics**  
✅ **Both notebook and script updated**  
✅ **Ready for training**

The evaluation will now work correctly during training! 🎉

## Next Steps

1. ✅ No code changes needed - fix is applied
2. ✅ Restart Kaggle kernel
3. ✅ Run all cells - metrics will compute correctly now

Your training will show proper accuracy and F1 scores during evaluation! 📊
