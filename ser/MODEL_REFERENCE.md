# Quick Model Reference Guide

## 🎯 Which Model Should I Use?

### For English Audio
✅ **Use**: nglish_model.h5
- Accuracy: 94.89%
- Path: cnn/models/saved_models/language_models/english/english_model.h5

### For Tamil Audio
✅ **Use**: 	amil_simple_model.h5
- Accuracy: 34.04%
- Path: cnn/models/saved_models/language_models/tamil/tamil_simple_model.h5

### For Sinhala Audio
⏳ **Not yet available** - Model needs to be trained

---

## 📦 All Saved Models

| Language | Model Name | Accuracy | Status | Use This? |
|----------|------------|----------|--------|-----------|
| English | nglish_model.h5 | 94.89% | Production | ✅ YES |
| Tamil | 	amil_simple_model.h5 | 34.04% | Best available | ✅ YES |
| Tamil | 	amil_transfer_model.h5 | 24.82% | Alternative | ❌ NO |
| Tamil | 	amil_model.h5 | 12.06% | Failed | ❌ NO |
| Sinhala | Not created yet | - | Pending | ⏳ PENDING |

---

## 🚀 Quick Usage

### Load and Use English Model
\\\python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('cnn/models/saved_models/language_models/english/english_model.h5')

# Predict (assuming preprocessed spectrogram)
predictions = model.predict(spectrogram)
\\\

### Load and Use Tamil Model
\\\python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('cnn/models/saved_models/language_models/tamil/tamil_simple_model.h5')

# Predict
predictions = model.predict(spectrogram)
\\\

---

## 📊 Model Training Scripts

| Model | Training Script | Preprocessing Script |
|-------|----------------|---------------------|
| English | 	rain.py | preprocess.py |
| Tamil Simple | 	rain_tamil_simple.py | preprocess_tamil.py |
| Tamil Transfer | 	rain_tamil_transfer.py | preprocess_tamil.py |
| Sinhala | TBD | TBD |

---

## 🔍 Testing Scripts

| Purpose | Script | Description |
|---------|--------|-------------|
| Test English | 	est_multiple.py | Test English model with samples |
| Test Tamil | 	est_tamil_model.py | Test Tamil model with samples |
| Compare Tamil | compare_tamil_models.py | Compare all 3 Tamil models |
| Test Bilingual | 	est_bilingual_extended.py | Test English+Tamil combined |

---

## 📝 Training Logs

| Model | Log File |
|-------|----------|
| English | 	raining_output.log |
| Tamil Original | 	raining_output_tamil.log |
| Tamil Simple | 	raining_simple_tamil.log |
| Tamil Transfer | 	raining_transfer_tamil.log |

---

## 🎯 Next Model to Train: SINHALA

**Recommended Approach**:
1. Add Sinhala dataset (~100 samples) to cnn/data/raw/SINHALA/
2. Apply data augmentation first (expand to ~900 samples)
3. Use Simple CNN architecture (like Tamil Simple)
4. Alternatively, use Transfer Learning from English model

**Expected Performance**: 40-50% accuracy (limited by small dataset)

---

**Quick Tip**: Always use the Simple CNN for languages with <1000 samples!
