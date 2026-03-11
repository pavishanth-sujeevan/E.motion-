# emotion2vec Fine-Tuning Plan

## 📋 Overview

**Goal:** Fine-tune emotion2vec_base pretrained model on English and Tamil datasets for 5-class emotion recognition (angry, fear, happy, neutral, sad).

**Pretrained Model:** emotion2vec_base
- Size: 1.07 GB
- Architecture: Self-supervised speech emotion representation
- Pretrained on: 160,000+ hours of speech data
- Universal: Works across languages and scenarios

**Expected Results:**
- English: 85-95% accuracy (vs current 94.89% CNN)
- Tamil: 65-80% accuracy (vs current 34.04% Simple CNN) ⚡ **HUGE IMPROVEMENT**

---

## 🎯 Three Approaches to Fine-Tuning

### **Approach 1: Feature Extraction (Fastest, Recommended First)**
**What:** Use emotion2vec as frozen feature extractor + train classifier head
**Pros:** 
- Fast training (5-10 mins)
- Requires less data
- No risk of catastrophic forgetting
**Cons:** 
- Limited adaptation to specific emotions
**Expected Accuracy:** English 80-85%, Tamil 60-70%

### **Approach 2: Linear Probing + Fine-tuning (Best Balance)**
**What:** First train classifier head, then fine-tune last few encoder layers
**Pros:** 
- Better adaptation to your emotions
- Controlled fine-tuning prevents overfitting
**Cons:** 
- Takes longer (20-30 mins)
**Expected Accuracy:** English 85-92%, Tamil 70-80% ⭐

### **Approach 3: End-to-End Fine-tuning (Maximum Performance)**
**What:** Fine-tune entire model with very low learning rate
**Pros:** 
- Maximum adaptation to your data
- Best possible accuracy
**Cons:** 
- Risk of overfitting on Tamil (only 936 samples)
- Slower training (40-60 mins)
**Expected Accuracy:** English 90-95%, Tamil 65-75%

---

## 📁 Implementation Structure

```
emotion2vec/
├── download_emotion2vec.py          [✅ DONE]
├── emotion2vec_base/                [✅ DOWNLOADED]
│   └── emotion2vec_base.pt (1.07GB)
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py        [Create: Extract features from audio]
│   ├── classifier.py                [Create: Emotion classifier head]
│   ├── dataset.py                   [Create: Load preprocessed data]
│   └── utils.py                     [Create: Helper functions]
├── scripts/
│   ├── 1_test_model.py              [Create: Verify model loads]
│   ├── 2_extract_features.py       [Create: Extract features from datasets]
│   ├── 3_train_classifier.py       [Create: Approach 1 - Feature extraction]
│   ├── 4_finetune_model.py         [Create: Approach 2 - Partial fine-tuning]
│   └── 5_full_finetune.py          [Create: Approach 3 - Full fine-tuning]
├── models/                          [Create: Save fine-tuned models]
│   ├── english_emotion2vec.pt
│   ├── tamil_emotion2vec.pt
│   └── checkpoints/
└── results/                         [Create: Training logs & comparison]
    ├── training_logs/
    └── comparison_with_cnn.md
```

---

## 🔧 Step-by-Step Implementation Plan

### **Phase 1: Setup & Verification** (5 mins)
1. ✅ Download emotion2vec model (DONE)
2. Create project structure
3. Test model loading with PyTorch
4. Extract features from sample audio
5. Verify feature dimensions

### **Phase 2: Data Preparation** (10 mins)
1. Load preprocessed audio from:
   - English: `cnn/data/processed_spectrograms/` (2,480 samples)
   - Tamil: `cnn/data/processed_tamil/` (936 samples)
2. Convert spectrograms back to audio (or use raw audio)
3. Extract emotion2vec features for all samples
4. Save features to disk for faster training

### **Phase 3: Approach 1 - Feature Extraction** (15 mins)
1. Freeze emotion2vec encoder
2. Train simple classifier head (2-3 layers)
3. Train on English → Evaluate
4. Train on Tamil → Evaluate
5. Compare with CNN results

### **Phase 4: Approach 2 - Partial Fine-tuning** (30 mins)
1. Load pretrained model
2. Train classifier head first (frozen encoder)
3. Unfreeze last 2-3 encoder layers
4. Fine-tune with low learning rate
5. Train English & Tamil separately
6. Evaluate and compare

### **Phase 5: Approach 3 - Full Fine-tuning** (Optional, 60 mins)
1. Fine-tune entire model end-to-end
2. Use very low learning rate (1e-5)
3. Heavy regularization (dropout, weight decay)
4. Only if Approach 2 shows promising results

### **Phase 6: Evaluation & Comparison** (15 mins)
1. Test all models on held-out test sets
2. Compare with existing CNN/LSTM models
3. Generate confusion matrices
4. Document results

---

## 🎓 Training Configuration

### **Approach 1: Feature Extraction**
```python
Model:
  - Frozen emotion2vec encoder
  - Dense(512) + ReLU + Dropout(0.3)
  - Dense(256) + ReLU + Dropout(0.3)
  - Dense(5) + Softmax

Training:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 50 (early stopping patience 10)
  - Loss: CrossEntropyLoss with class weights
```

### **Approach 2: Partial Fine-tuning**
```python
Phase 1 - Classifier Only:
  - Learning Rate: 0.001
  - Epochs: 20
  - Freeze all encoder layers

Phase 2 - Fine-tune Last Layers:
  - Learning Rate: 0.0001 (10x lower)
  - Epochs: 30
  - Unfreeze last 2-3 encoder layers
  - Early stopping patience 15
```

### **Approach 3: Full Fine-tuning**
```python
Model: Full emotion2vec
Training:
  - Learning Rate: 0.00001 (100x lower)
  - Batch Size: 16 (smaller for stability)
  - Epochs: 100
  - Gradient Clipping: 1.0
  - Weight Decay: 0.01
  - Early stopping patience 20
```

---

## 📊 Expected Timeline

| Phase | Duration | Priority |
|-------|----------|----------|
| Setup & Verification | 5 mins | High |
| Data Preparation | 10 mins | High |
| Approach 1 (Feature Extraction) | 15 mins | High ⭐ |
| Approach 2 (Partial Fine-tuning) | 30 mins | Medium ⭐⭐ |
| Approach 3 (Full Fine-tuning) | 60 mins | Low (Optional) |
| Evaluation & Comparison | 15 mins | High |
| **Total** | **2-2.5 hours** | |

---

## 🚀 Recommended Execution Order

### **Start Here: Quick Win Path** (40 mins total)
1. ✅ Download model (DONE)
2. Setup structure & test loading (5 mins)
3. Extract features from datasets (10 mins)
4. Train Approach 1 - English (10 mins)
5. Train Approach 1 - Tamil (10 mins)
6. Evaluate & compare (5 mins)

### **If Results Are Good: Advanced Path** (+45 mins)
7. Train Approach 2 - English (20 mins)
8. Train Approach 2 - Tamil (20 mins)
9. Final comparison (5 mins)

### **Only If Needed: Maximum Performance** (+60 mins)
10. Full end-to-end fine-tuning

---

## 💡 Key Technical Decisions

### **1. Audio Format**
- **Option A:** Use raw audio files (16kHz WAV) ← **RECOMMENDED**
  - emotion2vec expects raw audio
  - Best feature extraction
- **Option B:** Convert mel spectrograms back to audio
  - More complex, lossy conversion

### **2. Feature Extraction**
- Extract 768-dimensional embeddings per audio file
- Use utterance-level features (not frame-level)
- Cache features to disk for faster training iterations

### **3. Data Augmentation** (Tamil only)
- Time stretching (0.9x - 1.1x)
- Pitch shifting (±1 semitone)
- Add light background noise
- **Goal:** Boost Tamil from 936 → ~2,000 effective samples

### **4. Loss Function**
- Weighted CrossEntropy (handle class imbalance)
- Label Smoothing (0.1) for better generalization

---

## 🎯 Success Metrics

| Dataset | Current Best | Target (Approach 1) | Target (Approach 2) | Stretch Goal |
|---------|--------------|---------------------|---------------------|--------------|
| **English** | 94.89% (CNN) | 80-85% | 85-92% | 93-96% |
| **Tamil** | 34.04% (Simple CNN) | 60-70% ⚡ | 70-80% ⚡⚡ | 75-85% ⚡⚡⚡ |

**Improvement Targets:**
- Tamil: +35-45% absolute improvement (2x-2.5x better!)
- English: Maintain or slightly improve

---

## 🔍 Why emotion2vec Will Outperform CNN

1. **Pretrained on Massive Data:** 160k hours vs your 2.5k samples
2. **Self-Supervised Learning:** Learned universal emotion features
3. **Language Agnostic:** Trained on multiple languages
4. **Better Features:** Deep representations vs hand-crafted mel-spectrograms
5. **Transfer Learning:** Leverages knowledge from millions of audio samples

---

## 📝 Next Steps

**Ready to start?** I'll create:
1. Test script to verify model loads
2. Feature extraction pipeline
3. Training script for Approach 1 (Feature Extraction)

Then we can train English & Tamil and compare results with your current CNNs!

**Estimated time to first results:** 30-40 minutes

Should I proceed with implementation? 🚀
