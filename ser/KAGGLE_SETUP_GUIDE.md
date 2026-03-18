# Kaggle Multilingual SER - Setup Guide

## 📋 Quick Start Instructions

### Step 1: Create Kaggle Notebook
1. Go to https://www.kaggle.com/
2. Click **"New Notebook"**
3. Title it: "Multilingual Speech Emotion Recognition with DAPT"

### Step 2: Enable GPU
1. Click **Settings** (right sidebar)
2. Under **Accelerator**, select **"GPU T4 x2"** or **"GPU P100"**
3. Click **Save**

### Step 3: Add Datasets
Add these public datasets to your notebook:

**Required:**
- **RAVDESS**: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
- **TESS**: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

**Optional (Tamil):**
- If you have EmoTa dataset, upload it as a Kaggle dataset
- Otherwise, the script will use a subset of English data as placeholder

**To add datasets:**
1. Click **"+ Add Data"** (right sidebar)
2. Search for "RAVDESS" → Click **"Add"**
3. Search for "TESS" → Click **"Add"**

### Step 4: Copy & Paste Code
1. Open the file: `kaggle_multilingual_ser.py`
2. Select ALL content (Ctrl+A)
3. Copy (Ctrl+C)
4. In Kaggle notebook, create a new code cell
5. Paste (Ctrl+V)

### Step 5: Run
1. Click **"Run All"** or press **Shift+Enter**
2. Wait 4-6 hours for training to complete
3. Download your trained model from `/kaggle/working/multilingual_ser/final_model/`

---

## 🎯 What the Script Does

### Phase 1: Dataset Preparation (5-10 min)
- Auto-detects RAVDESS and TESS datasets
- Creates train/validation splits
- Prepares mock Tamil and Sinhala datasets

### Phase 2: Domain-Adaptive Pre-training (30-60 min)
- Pre-trains on 500 unlabeled Sinhala samples
- Adapts the model to new domain
- 5 epochs of contrastive learning

### Phase 3: Multilingual Fine-tuning (3-5 hours)
- Trains on English, Tamil, and Sinhala simultaneously
- Uses weighted sampling to balance languages
- Applies audio augmentation (pitch shift, time stretch, room impulse)
- 20 epochs with early stopping

### Phase 4: Save & Evaluate
- Saves final model to `/kaggle/working/`
- Computes accuracy and F1-score
- Ready for download and inference

---

## 🔧 Configuration Options

You can modify these parameters in the `Config` class:

```python
@dataclass
class Config:
    # Model
    model_name: str = "facebook/wav2vec2-xls-r-300m"  # Change to wav2vec2-base for faster training
    num_labels: int = 7  # Number of emotions
    freeze_layers: int = 12  # Number of frozen transformer layers
    
    # Training
    batch_size: int = 16  # Reduce to 8 if GPU memory is low
    num_epochs: int = 20  # Increase for better accuracy
    
    # DAPT
    dapt_epochs: int = 5  # Domain adaptation epochs
    
    # Learning rates
    lr_backbone: float = 1e-5  # Backbone learning rate
    lr_head: float = 1e-4  # Classification head learning rate
```

---

## 💾 Expected Output

After training completes, you'll find:

```
/kaggle/working/multilingual_ser/
├── dapt_checkpoint/          # After DAPT phase
│   ├── config.json
│   └── pytorch_model.bin
├── checkpoint-*/             # Training checkpoints
└── final_model/              # 👈 DOWNLOAD THIS
    ├── config.json
    ├── pytorch_model.bin
    ├── preprocessor_config.json
    └── training_args.bin
```

---

## 📊 Expected Performance

Based on the architecture and datasets:

| Language | Expected Accuracy | Expected F1 |
|----------|-------------------|-------------|
| English  | 70-80%           | 0.68-0.78   |
| Tamil    | 65-75%           | 0.63-0.73   |
| Sinhala  | 60-70%           | 0.58-0.68   |

*Note: Sinhala is mock data, so results are illustrative*

---

## 🚀 Using the Trained Model

After downloading the model, you can use it for inference:

```python
from transformers import Wav2Vec2FeatureExtractor
import torch
import librosa

# Load model
model = Wav2Vec2ForSER.from_pretrained("path/to/final_model")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("path/to/final_model")
model.eval()

# Load audio
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Extract features
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs['logits'], dim=-1)

# Emotion mapping
emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
predicted_emotion = emotions[predictions.item()]

print(f"Predicted Emotion: {predicted_emotion}")
```

---

## ⚡ Performance Tips

### For Faster Training:
1. **Use smaller model**: Change to `wav2vec2-base` (95M params)
   ```python
   model_name: str = "facebook/wav2vec2-base"
   ```

2. **Reduce batch size**: If GPU memory is limited
   ```python
   batch_size: int = 8
   ```

3. **Skip DAPT**: Comment out DAPT section to save 30-60 min

4. **Fewer epochs**: Reduce to 10-15 epochs
   ```python
   num_epochs: int = 10
   ```

### For Better Accuracy:
1. **More epochs**: Increase to 30-40
2. **Lower learning rate**: Try `lr_backbone: 5e-6`
3. **Add more data**: Upload your own datasets
4. **Adjust augmentation**: Modify `AudioAugmenter` parameters

---

## 🐛 Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce batch size
```python
batch_size: int = 8  # or even 4
```

### "Dataset not found"
**Solution**: Check dataset names in Kaggle
```python
# The script auto-detects, but you can manually set:
RAVDESS_PATH = "/kaggle/input/ravdess-emotional-speech-audio"
TESS_PATH = "/kaggle/input/toronto-emotional-speech-set-tess"
```

### "Training is too slow"
**Solution**: 
1. Make sure GPU is enabled (not CPU)
2. Use smaller model: `wav2vec2-base`
3. Reduce `max_duration` to 6.0 seconds

### "Model not saving"
**Solution**: 
- Kaggle has limited disk space
- Delete checkpoint folders periodically:
```python
save_total_limit=2  # Keep only 2 checkpoints
```

---

## 📚 Additional Resources

- **Wav2Vec2 Paper**: https://arxiv.org/abs/2006.11477
- **XLS-R Paper**: https://arxiv.org/abs/2111.09296
- **Hugging Face Docs**: https://huggingface.co/docs/transformers/model_doc/wav2vec2

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{babu2021xlsr,
  title={XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale},
  author={Babu, Arun and Wang, Changhan and Tjandra, Andros and others},
  journal={arXiv preprint arXiv:2111.09296},
  year={2021}
}
```

---

## 📝 License

This code is provided as-is for educational and research purposes.

---

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Kaggle notebook output logs
3. Verify GPU is enabled
4. Ensure datasets are properly added

---

**Happy Training! 🚀**
