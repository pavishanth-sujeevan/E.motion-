# E.motion - Speech Emotion Recognition System

A multilingual speech emotion recognition system using CNN models trained on mel spectrograms. Supports English, Tamil, and Sinhala languages with separate language-specific models.

## 📋 Project Overview

This project aims to recognize 5 emotions from speech audio:
- **Angry**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**

**Architecture**: Convolutional Neural Networks (CNN) trained on mel spectrogram features extracted from 3-second audio clips.

---

## 🎯 Trained Models & Performance

### English Model ✅ PRODUCTION READY

**Model File**: `cnn/models/saved_models/language_models/english/english_model.h5`

**Performance**:
- **Test Accuracy**: 94.89%
- **Training Samples**: 2,480 (RAVDESS + TESS)
- **Architecture**: Deep CNN, 1.27M parameters
- **Status**: ✅ Production ready

**Per-Emotion Accuracy**:
| Emotion | Accuracy |
|---------|----------|
| Angry   | 97.3%    |
| Fear    | 87.8%    |
| Happy   | 95.9%    |
| Neutral | 98.7%    |
| Sad     | 94.6%    |

**Datasets Used**:
- RAVDESS (480 samples after filtering)
- TESS (2,000 samples)

---

### Tamil Models

#### 1. Tamil Simple CNN ✅ BEST TAMIL MODEL

**Model File**: `cnn/models/saved_models/language_models/tamil/tamil_simple_model.h5`

**Performance**:
- **Test Accuracy**: 34.04%
- **Training Samples**: 936 (EMOTA)
- **Architecture**: Simple CNN, 118K parameters (10x fewer)
- **Status**: ⚠️ Best available, but needs improvement

**Per-Emotion Accuracy**:
| Emotion | Accuracy |
|---------|----------|
| Angry   | 23.3%    |
| Fear    | 64.7%    |
| Happy   | 51.6%    |
| Neutral | 16.1%    |
| Sad     | 28.1%    |

**Why This Works Best**:
- Fewer parameters (118K) prevent overfitting on small dataset
- Better generalization across all emotions
- 2.8x improvement over deep CNN

---

#### 2. Tamil Transfer Learning Model

**Model File**: `cnn/models/saved_models/language_models/tamil/tamil_transfer_model.h5`

**Performance**:
- **Test Accuracy**: 24.82%
- **Training Samples**: 936 (EMOTA)
- **Architecture**: Deep CNN with frozen conv layers, 1.27M parameters
- **Status**: ⚠️ Moderate performance

**Per-Emotion Accuracy**:
| Emotion | Accuracy |
|---------|----------|
| Angry   | 13.3%    |
| Fear    | 17.6%    |
| Happy   | 77.4%    |
| Neutral | 3.2%     |
| Sad     | 9.4%     |

**Note**: Biased toward happy emotion. English features do not transfer perfectly to Tamil.

---

#### 3. Tamil Original Deep CNN ❌ FAILED

**Model File**: `cnn/models/saved_models/language_models/tamil/tamil_model.h5`

**Performance**:
- **Test Accuracy**: 12.06%
- **Training Samples**: 936 (EMOTA)
- **Architecture**: Deep CNN, 1.27M parameters
- **Status**: ❌ Failed - Severe overfitting

**Per-Emotion Accuracy**:
| Emotion | Accuracy |
|---------|----------|
| Angry   | 0.0%     |
| Fear    | 100.0%   |
| Happy   | 0.0%     |
| Neutral | 0.0%     |
| Sad     | 0.0%     |

**Problem**: Too many parameters for 936 samples. Model learned to predict only fear.

---

### Sinhala Model 🔜 PENDING

**Model File**: Not yet created

**Status**: ⏳ Awaiting training
- **Expected Samples**: ~100
- **Challenge**: Very limited data
- **Recommended Approach**: Transfer learning or data augmentation

---

## 📊 Model Comparison Summary

| Model | File Name | Accuracy | Samples | Parameters | Status |
|-------|-----------|----------|---------|------------|--------|
| English Deep CNN | `english_model.h5` | **94.89%** | 2,480 | 1.27M | ✅ Production |
| Tamil Simple CNN | `tamil_simple_model.h5` | **34.04%** | 936 | 118K | ⚠️ Best Tamil |
| Tamil Transfer | `tamil_transfer_model.h5` | 24.82% | 936 | 1.27M | ⚠️ Moderate |
| Tamil Deep CNN | `tamil_model.h5` | 12.06% | 936 | 1.27M | ❌ Failed |
| Sinhala | TBD | TBD | ~100 | TBD | ⏳ Pending |

---

## 🗂️ Dataset Information

### RAVDESS (English)
- **Path**: `cnn/data/raw/RAVDESS-SPEECH/`
- **Total Files**: 1,440
- **Used Files**: 480 (after filtering)
- **Emotions**: 5 (angry, fear, happy, neutral, sad)
- **Filtering Criteria**:
  - Emotion codes: 01, 03, 04, 05, 06
  - Intensity: 01 for neutral, 02 for others
  - Statements: 01, 02 only
  - Repetitions: 01, 02 only

### TESS (English)
- **Path**: `cnn/data/raw/TESS/TESS Toronto emotional speech set data/`
- **Total Files**: 2,000
- **Emotions**: 5 (angry, fear, happy, neutral, sad)

### EMOTA (Tamil)
- **Path**: `cnn/data/raw/EMOTA/TamilSER-DB/`
- **Total Files**: 936
- **Emotions**: 5 (angry, fear, happy, neutral, sad)
- **Format**: `{speaker}_{utterance}_{emotion}.wav`

### Sinhala Dataset
- **Path**: `cnn/data/raw/SINHALA/` (when added)
- **Total Files**: ~100
- **Status**: Not yet added

---

## 🛠️ Technical Details

### Model Architecture

**Deep CNN** (English, Original Tamil):
- 4 Convolutional Blocks (32 → 64 → 128 → 256 filters)
- Batch Normalization + Dropout (0.4-0.5)
- Global Average Pooling
- 2 Dense Layers (256 → 128 → 5)
- **Total Parameters**: 1,276,389

**Simple CNN** (Best Tamil):
- 3 Convolutional Blocks (32 → 64 → 128 filters)
- Batch Normalization + Dropout (0.3-0.5)
- Global Average Pooling
- 2 Dense Layers (128 → 64 → 5)
- **Total Parameters**: 118,661

### Audio Processing
- **Sample Rate**: 22,050 Hz
- **Duration**: 3 seconds
- **Feature**: Mel Spectrogram
  - N_MELS: 128
  - N_FFT: 2048
  - HOP_LENGTH: 512
  - Max Time Steps: 130
- **Input Shape**: (128, 130, 1)

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (0.0001 for transfer learning)
- **Batch Size**: 16
- **Epochs**: 50 (with early stopping)
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**:
  - Early Stopping (patience: 20-25)
  - ReduceLROnPlateau (factor: 0.5, patience: 8-10)
  - ModelCheckpoint (save best only)

---

## 📁 Project Structure

```
E.motion-/
├── cnn/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── RAVDESS-SPEECH/
│   │   │   ├── TESS/
│   │   │   ├── EMOTA/
│   │   │   └── SINHALA/ (to be added)
│   │   ├── processed_spectrograms/ (English + Tamil combined)
│   │   └── processed_tamil/ (Tamil only)
│   │
│   ├── models/
│   │   └── saved_models/
│   │       └── language_models/
│   │           ├── english/
│   │           │   └── english_model.h5 (94.89%)
│   │           ├── tamil/
│   │           │   ├── tamil_simple_model.h5 (34.04%) ⭐ BEST
│   │           │   ├── tamil_transfer_model.h5 (24.82%)
│   │           │   └── tamil_model.h5 (12.06%)
│   │           └── sinhala/ (pending)
│   │
│   └── src/
│       ├── config.py (configuration)
│       ├── preprocess.py (multilingual preprocessing)
│       ├── preprocess_tamil.py (Tamil-only preprocessing)
│       ├── train.py (deep CNN training)
│       ├── train_tamil_simple.py (simple CNN training)
│       ├── train_tamil_transfer.py (transfer learning)
│       ├── compare_tamil_models.py (model comparison)
│       ├── test_tamil_model.py (testing script)
│       └── predict_audio.py (inference)
│
├── emotion2vec/ (separate emotion recognition approach)
├── README.md (this file)
├── TAMIL_MODEL_RESULTS.md (detailed Tamil results)
└── .gitignore
```

---

## 🚀 Usage

### Testing Models

**Test English Model**:
```bash
cd cnn/src
python test_multiple.py
```

**Test Tamil Model**:
```bash
cd cnn/src
python test_tamil_model.py
```

**Compare All Tamil Models**:
```bash
cd cnn/src
python compare_tamil_models.py
```

### Training New Models

**Train English Model**:
```bash
cd cnn/src
python preprocess.py  # Preprocess data first
python train.py       # Train model
```

**Train Tamil Simple Model**:
```bash
cd cnn/src
python preprocess_tamil.py  # Preprocess Tamil data
python train_tamil_simple.py  # Train simple CNN
```

**Train Tamil Transfer Learning Model**:
```bash
cd cnn/src
python preprocess_tamil.py  # Preprocess Tamil data
python train_tamil_transfer.py  # Transfer learning from English
```

---

## 📈 Training Results Summary

### Training Logs Location
- English: `cnn/training_output.log`
- Tamil Original: `cnn/training_output_tamil.log`
- Tamil Simple: `cnn/training_simple_tamil.log`
- Tamil Transfer: `cnn/training_transfer_tamil.log`

### Training Time
- **English Model**: ~60 minutes (50 epochs, early stopped at ~35)
- **Tamil Simple**: ~10 minutes (50 epochs, early stopped at ~15)
- **Tamil Transfer**: ~20 minutes (50 epochs, early stopped at ~25)

---

## 🎯 Next Steps & Recommendations

### For Tamil Model Improvement

Current best Tamil accuracy is only 34.04%. To improve:

#### Option 1: Data Augmentation (RECOMMENDED)
- Pitch shifting (±2-3 semitones) → 5x data
- Time stretching (0.85x-1.15x speed) → 3x data
- Background noise injection → 2x data
- **Expected Result**: Expand 936 → 8,000+ samples
- **Expected Accuracy**: 55-65%

#### Option 2: Collect More Data
- Target: 2,000-3,000 Tamil samples minimum
- Ensure balanced distribution across 5 emotions
- Multiple speakers for better generalization

#### Option 3: Ensemble Methods
- Combine predictions from multiple models
- Weighted voting based on confidence
- Expected improvement: +2-5%

### For Sinhala Model

With only ~100 samples, recommended approaches:

1. **Transfer Learning** from English model (freeze more layers)
2. **Aggressive Data Augmentation** to expand dataset
3. **Simplest CNN** possible (even fewer parameters than Tamil Simple)
4. **Realistic Expectations**: 40-50% accuracy at best

---

## 📊 Key Insights

### What Worked

1. **English Model (94.89%)**
   - Large dataset (2,480 samples) is key
   - Deep architecture works well with sufficient data
   - Balanced emotion distribution

2. **Tamil Simple CNN (34.04%)**
   - Fewer parameters prevent overfitting
   - Better than transfer learning for limited data
   - Simpler is better when data is scarce

### What Didnt Work

1. **Deep CNN on Small Dataset (12.06%)**
   - 1.27M parameters too many for 936 samples
   - Severe overfitting
   - Learned to predict only one emotion

2. **Transfer Learning for Tamil (24.82%)**
   - English features do not transfer perfectly
   - Language-specific acoustic features differ
   - Frozen layers may be too rigid

### Critical Success Factors

| Factor | English ✅ | Tamil ⚠️ | Sinhala ❓ |
|--------|-----------|---------|-----------|
| Dataset Size | 2,480 | 936 | ~100 |
| Quality | High | Moderate | Unknown |
| Balance | Good | Good | Unknown |
| Model Complexity | Deep (1.27M) | Simple (118K) | TBD |
| Result | 94.89% | 34.04% | TBD |

**Lesson**: Dataset size is the most critical factor. Need 2,000+ samples per language for good performance.

---

## 🔧 Configuration

All configuration is centralized in `cnn/src/config.py`:

```python
# Target emotions
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
MAX_TIME_STEPS = 130

# Training parameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
```

---

## 📝 Notes

- All models use mel spectrograms as input features
- Models are saved in HDF5 format (.h5)
- Use `tamil_simple_model.h5` for Tamil predictions
- Use `english_model.h5` for English predictions
- Sinhala model not yet trained

---

## 🤝 Contributing

When training new models:
1. Use consistent preprocessing (config.py parameters)
2. Save models to appropriate language folder
3. Document performance in this README
4. Update model comparison table
5. Save training logs with descriptive names

---

## 📄 License

[Add license information]

---

## 👥 Authors

[Add author information]

---

## 📚 References

### Datasets
- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- TESS: Toronto Emotional Speech Set
- EMOTA: Tamil Speech Emotion Recognition Database

### Papers
- [Add relevant papers]

---

**Last Updated**: 2026-02-20

**Status**: English model production-ready. Tamil model needs improvement. Sinhala model pending.
