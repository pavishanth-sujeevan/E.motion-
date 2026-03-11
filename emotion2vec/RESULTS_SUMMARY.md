# emotion2vec Feature-Based Classifier Results

**Date:** 2026-02-21  
**Approach:** Feature extraction from spectrograms + MLP classifier

## Results Summary

| Language | Test Accuracy | vs CNN | Status |
|----------|---------------|--------|--------|
| **English** | **73.68%** | -21.21% (CNN: 94.89%) | ❌ Worse |
| **Tamil** | **34.04%** | +0.00% (Simple CNN: 34.04%) | ➡️ Same |

---

## Analysis

### English Performance (73.68%)
**Significantly worse than CNN (94.89%)**

**Per-Class Performance:**
- Fear: 87.91% ✓ (Best)
- Angry: 78.85% ✓
- Neutral: 72.64%
- Sad: 72.64%
- Happy: 58.49% ⚠️ (Worst)

**Why worse than CNN?**
1. Simple statistical features from spectrograms (100-dim) vs deep CNN learned features
2. No proper emotion2vec embeddings - just MFCC-like statistics
3. MLP classifier is simpler than 4-layer CNN

### Tamil Performance (34.04%)
**Same as Simple CNN**

**Per-Class Performance:**
- Fear: 82.35% ✓ (Excellent!)
- Sad: 46.88%
- Angry: 33.33%
- Neutral: 19.35% ⚠️
- Happy: 9.68% ❌ (Very poor)

**Why same as CNN?**
- Both models struggle with limited data (654 training samples)
- Feature quality bottleneck more than architecture
- Similar per-class bias patterns (fear performs well, happy/neutral struggle)

---

## Key Insights

### ❌ Current Approach Limitations

1. **Not using real emotion2vec embeddings**
   - We extracted simple statistical features from spectrograms
   - emotion2vec model requires proper FairSeq integration
   - Current features are essentially advanced MFCCs

2. **Missing the key advantage**
   - emotion2vec's power is in its pretrained representations
   - We didn't actually use the pretrained model for feature extraction
   - Just trained a simple MLP on hand-crafted features

3. **No transfer learning benefit**
   - Didn't leverage 160k hours of pretrained data
   - Essentially a baseline statistical model

### ✅ What Worked

1. **Fast training** (< 5 mins total)
2. **Good Fear recognition** (82-88% on both languages)
3. **Established baseline** for comparison

---

## Why This Doesn't Beat CNN

The emotion2vec_base.pt file we downloaded is a **FairSeq checkpoint** that requires:
- FairSeq framework installation
- Specific model architecture loading
- Proper forward pass through encoder layers
- Feature extraction from encoder outputs

Our current approach:
- ❌ Loads the checkpoint but doesn't use it
- ❌ Extracts features from spectrograms directly
- ❌ Trains simple MLP (no pretrained knowledge)
- ✓ Fast but limited performance

---

## Next Steps Options

### Option A: Properly Integrate emotion2vec ⭐ (Recommended)
**Time:** 2-3 hours  
**Expected Improvement:** Tamil 34% → 60-75%

1. Install FairSeq framework
2. Load emotion2vec encoder properly
3. Extract 768-dim pretrained embeddings
4. Fine-tune on your data

**Pros:**
- Will actually use pretrained knowledge
- Significant Tamil improvement expected
- Proper transfer learning

**Cons:**
- Requires FairSeq (complex dependencies)
- More implementation work

### Option B: Stick with CNN Models ✅ (Practical)
**Current Status:** 
- English: CNN 94.89% (excellent!)
- Tamil: Simple CNN 34.04% (baseline)

**Why this makes sense:**
- CNN already achieving 95% on English
- emotion2vec integration is complex
- Tamil needs more data, not just better features

**Action:** Accept CNN results and focus on data collection for Tamil

### Option C: Try Data Augmentation on Tamil ⚡ (Quick Win)
**Time:** 1 hour  
**Expected:** Tamil 34% → 45-55%

1. Apply audio augmentation to Tamil
2. Retrain Simple CNN
3. Likely easier than emotion2vec integration

---

## Recommendation

**For your use case:**

1. **Keep English CNN** (94.89%) - It's already excellent! ✅

2. **Tamil has 3 paths:**
   - **Path A (Quick):** Data augmentation → Expected 45-55%
   - **Path B (Complex):** Proper emotion2vec integration → Expected 60-75%
   - **Path C (Long-term):** Collect more Tamil data → Expected 70-85%

3. **Most practical:** Accept Tamil at 34% for now, or try augmentation first

---

## Technical Notes

**Model Architecture Used:**
```
Input (100-dim features)
↓
Dense(256) + ReLU + Dropout(0.3) + BatchNorm
↓
Dense(128) + ReLU + Dropout(0.3) + BatchNorm
↓
Dense(64) + ReLU + Dropout(0.2)
↓
Dense(5) [Output]
```

**Parameters:** 68,101  
**Training Time:** 
- English: ~2 mins (25 epochs)
- Tamil: ~3 mins (46 epochs)

**Feature Extraction:**
- Mean, Std, Max, Min of spectrograms
- 96 sampled spectrogram values
- Total: 100-dimensional feature vector

---

## Conclusion

This approach **did not improve** over existing CNN models because:
1. We didn't properly use emotion2vec's pretrained representations
2. Simple statistical features ≈ basic MFCCs (not as powerful as deep features)
3. MLP classifier < CNN for this task

**Bottom Line:**
- English: Stick with CNN (94.89%) ✅
- Tamil: Either accept 34% or try data augmentation first
- emotion2vec proper integration would require significant additional work
