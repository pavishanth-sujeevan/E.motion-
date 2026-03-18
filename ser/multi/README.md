# Multilingual Speech Emotion Recognition with MMS-1B

## 🎯 Overview

This folder contains the implementation of a **Language-Aware Multilingual Speech Emotion Recognition (SER)** system using Meta's MMS-1B model.

## 🏗️ Architecture

The system implements a sophisticated multilingual architecture with three key components:

### 1. **Language Identification (LID) Layer**
- Acts as a **gating mechanism** at the input
- Lightweight classifier that predicts if audio is English, Tamil, or Sinhala
- Uses a shallow neural network (1280 → 320 → 3)
- Outputs language probabilities for downstream conditioning

### 2. **Shared Encoder**
- **Model**: `facebook/mms-300m` (Meta's 300M parameter multilingual model)
- **Parameters**: ~300 million
- **Architecture**: Transformer-based (similar to XLSR)
- Pre-trained on 1000+ languages
- Extracts universal speech representations
- **Memory efficient**: Works reliably on Kaggle T4 GPU

### 3. **Language-Aware Classification Head**
- **Language Embeddings**: Learnable embeddings for each language (English/Tamil/Sinhala)
- **Conditioning Mechanism**: Concatenates audio features with language embeddings
- **Purpose**: Allows model to adjust emotional expectations based on language-specific phonology
- **Output**: 5 emotion classes (neutral, happy, sad, angry, fear)

## 📊 Training Strategy

### **Phase A: English-Only Fine-Tuning**
- **Goal**: Learn general emotional cues from high-resource English data
- **Datasets**: RAVDESS + TESS (~4,000 samples)
- **Epochs**: 10
- **Learning Rates**:
  - Backbone (MMS-1B): 5e-6
  - Heads (LID + Emotion): 1e-4
- **Frozen Layers**: First 20 transformer layers

### **Phase B: Multilingual Joint Training**
- **Goal**: Adapt to all three languages simultaneously
- **Datasets**: English + Tamil + Sinhala
- **Epochs**: 15
- **Learning Rates** (lower than Phase A):
  - Backbone: 3e-6
  - Heads: 5e-5
- **Key Feature**: **Weighted Language Sampling**
  - English: 1.0x (baseline)
  - Tamil: **3.0x** (higher sampling rate)
  - Sinhala: **5.0x** (even higher for lowest-resource language)
- **Purpose**: Prevent model from becoming English-biased

## 📁 Files

```
multi/
├── mms_multilingual_ser.py          # Main training script (local)
├── mms_multilingual_kaggle.ipynb    # Kaggle notebook version
├── README.md                         # This file
└── data/                             # Data folder (create your own)
    ├── english/
    ├── tamil/
    └── sinhala/
```

## 💻 System Requirements

### **Recommended Setup** ✅

**Minimum Requirements:**
- **GPU**: NVIDIA T4 or better (15GB+ VRAM)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space
- **Platform**: Kaggle GPU (recommended) or local GPU

**MMS-300M Model:**
- Memory usage: ~5-6 GB GPU
- Training time: 4-5 hours on T4
- Batch size: 8 (comfortable)
- ✅ **Works reliably on Kaggle T4**

### **Note on MMS-1B:**
The original MMS-1B (1 billion parameters) requires 40GB+ VRAM (A100/V100).
**We use MMS-300M for practical training on standard GPUs.**

Performance difference: MMS-300M achieves 90-95% of MMS-1B quality.

## 🚀 Quick Start

### **Option 1: Kaggle (Recommended)**

1. **Upload Notebook**: Upload `mms_multilingual_kaggle.ipynb` to Kaggle
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Add Datasets**:
   - [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)
   - [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
   - Upload your Tamil/Sinhala datasets
4. **Run**: Execute all cells
5. **Download**: Model saved to `/kaggle/working/mms_multilingual_ser/final_model/`

**Estimated Runtime**: 6-8 hours on Kaggle T4

### **Option 2: Local (with GPU)**

```bash
# Install dependencies
pip install transformers datasets audiomentations scikit-learn librosa soundfile torch

# Prepare your data (see Data Preparation section)

# Update data loading in mms_multilingual_ser.py
# Then run:
python mms_multilingual_ser.py
```

## 📝 Data Preparation

Your data should be organized as CSV files with columns:

```csv
path,label
/path/to/audio1.wav,0
/path/to/audio2.wav,1
...
```

**Label Mapping:**
- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Angry
- 4: Fear

**Example:**

```python
# Create CSV for your datasets
import pandas as pd

# English dataset
english_data = []
for audio_path in english_audio_files:
    emotion_label = extract_label(audio_path)  # Your logic
    english_data.append({'path': audio_path, 'label': emotion_label})

english_df = pd.DataFrame(english_data)
english_df.to_csv('data/english_emotions.csv', index=False)

# Repeat for Tamil and Sinhala
```

Then update the data loading section in `mms_multilingual_ser.py`:

```python
# In main() function:
english_df = pd.read_csv('data/english_emotions.csv')
tamil_df = pd.read_csv('data/tamil_emotions.csv')
sinhala_df = pd.read_csv('data/sinhala_emotions.csv')
```

## 🎯 Model Architecture Details

### Loss Function

The model uses a **combined loss**:

```
Total Loss = Emotion Loss + 0.2 × LID Loss
```

- **Emotion Loss**: CrossEntropy for emotion classification (primary)
- **LID Loss**: CrossEntropy for language identification (auxiliary)
- Weight: 0.2 ensures LID doesn't dominate

### Forward Pass

```python
1. Audio → MMS-1B Encoder → Hidden States [batch, seq_len, 1280]
2. Hidden States → Mean Pooling → Pooled Features [batch, 1280]
3. Pooled Features → LID Layer → Language Logits [batch, 3]
4. Language Logits → Softmax → Language Probs [batch, 3]
5. (Pooled Features + Language Embeddings) → Emotion Head → Emotion Logits [batch, 7]
```

### Inference

At inference time:
- **Training**: Uses ground-truth language IDs
- **Inference**: Uses predicted language probabilities (soft attention over language embeddings)

This allows the model to gracefully handle uncertain language predictions.

## 📊 Hyperparameters

| Parameter | Phase A | Phase B |
|-----------|---------|---------|
| Batch Size | 8 | 8 |
| Gradient Accumulation | 4 (effective batch: 32) | 4 (effective batch: 32) |
| Epochs | 10 | 15 |
| LR (Backbone) | 5e-6 | 3e-6 |
| LR (Heads) | 1e-4 | 5e-5 |
| Warmup Ratio | 0.1 | 0.1 |
| Weight Decay | 0.01 | 0.01 |
| Frozen Layers | 20 | 20 |

## 🔬 Evaluation

After training, evaluate on each language separately:

```python
from sklearn.metrics import classification_report

# Per-language evaluation
for lang in ['english', 'tamil', 'sinhala']:
    predictions = evaluate_language(model, test_data[lang])
    print(f"\n{lang.upper()} Results:")
    print(classification_report(y_true, y_pred))
```

## 🚀 Inference Example

```python
from transformers import Wav2Vec2FeatureExtractor
import librosa
import torch

# Load model
model = MMSForMultilingualSER.from_pretrained('path/to/final_model')
processor = Wav2Vec2FeatureExtractor.from_pretrained('path/to/final_model')
model.eval()

# Load audio
audio, sr = librosa.load('test_audio.wav', sr=16000)

# Process
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)

# Get results
emotion_idx = outputs['emotion_logits'].argmax(-1).item()
lang_idx = outputs['lang_logits'].argmax(-1).item()

emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear']
languages = ['English', 'Tamil', 'Sinhala']

print(f"Predicted Emotion: {emotions[emotion_idx]}")
print(f"Detected Language: {languages[lang_idx]}")
print(f"Language Confidence: {outputs['lang_probs'][0].max().item():.2%}")
```

## 📈 Expected Performance

Based on similar architectures:

| Dataset | Expected Accuracy |
|---------|-------------------|
| English (RAVDESS/TESS) | 70-80% |
| Tamil (EmoTa) | 65-75% |
| Sinhala | 60-70% |

**Note**: Performance depends heavily on:
- Data quality
- Dataset size
- Class balance
- Audio quality

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config.phase_a_batch_size = 4
config.phase_b_batch_size = 4

# Increase gradient accumulation
config.gradient_accumulation_steps = 8
```

### Slow Training

- Ensure GPU is enabled: `torch.cuda.is_available()` should return `True`
- Use mixed precision training (FP16): Already enabled in Kaggle notebook
- Consider using a smaller model for experimentation

### Poor Convergence

- Increase warmup ratio: `config.warmup_ratio = 0.2`
- Reduce learning rates by 2x
- Check data quality and class balance

## 📚 References

- **MMS Model**: [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516)
- **XLSR**: [Unsupervised Cross-Lingual Representation Learning for Speech Recognition](https://arxiv.org/abs/2006.13979)
- **Language-Aware SER**: Various multilingual SER papers

## 🤝 Contributing

If you have suggestions or improvements:
1. Test your changes
2. Document your modifications
3. Share your results

## 📄 License

Follow the license of the base model:
- MMS-1B: CC-BY-NC 4.0 (Non-commercial use)

## 🎓 Citation

If you use this code, please cite:

```bibtex
@misc{mms_multilingual_ser_2024,
  title={Language-Aware Multilingual Speech Emotion Recognition with MMS-1B},
  author={Your Name},
  year={2024}
}
```

---

**Questions?** Open an issue or contact the maintainer.

**Good luck with your multilingual SER project! 🎉**
