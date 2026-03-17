"""
Test emotion-specific pretrained models on English and Tamil data:
1. speechbrain/emotion-recognition-wav2vec2-IEMOCAP (most popular)
2. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
3. Rajaram1996/Hubert_emotion
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, HubertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}
TARGET_SR = 16000

def collect_english_files():
    """Collect English audio files"""
    data_root = Path('../data/raw')
    files = []
    labels = []
    
    # RAVDESS
    ravdess_path = data_root / 'RAVDESS-SPEECH'
    if ravdess_path.exists():
        for audio_file in ravdess_path.rglob('*.wav'):
            parts = audio_file.stem.split('-')
            emotion_code = int(parts[2])
            emotion_map = {3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 1: 'neutral'}
            if emotion_code in emotion_map:
                files.append(str(audio_file))
                labels.append(EMOTION_TO_INDEX[emotion_map[emotion_code]])
    
    # TESS
    tess_path = data_root / 'TESS'
    if tess_path.exists():
        for audio_file in tess_path.rglob('*.wav'):
            filename = audio_file.stem.lower()
            if 'angry' in filename:
                emotion = 'angry'
            elif 'fear' in filename:
                emotion = 'fear'
            elif 'happy' in filename or 'ps' in filename:
                emotion = 'happy'
            elif 'neutral' in filename:
                emotion = 'neutral'
            elif 'sad' in filename:
                emotion = 'sad'
            else:
                continue
            files.append(str(audio_file))
            labels.append(EMOTION_TO_INDEX[emotion])
    
    return files, labels

def collect_tamil_files():
    """Collect Tamil audio files"""
    data_root = Path('../data/raw/EmoTa/TamilSER-DB')
    files = []
    labels = []
    
    if not data_root.exists():
        return files, labels
    
    for audio_file in data_root.rglob('*.wav'):
        filename = audio_file.stem.lower()
        
        if 'angry' in filename or 'ang' in filename:
            emotion = 'angry'
        elif 'fear' in filename:
            emotion = 'fear'
        elif 'happy' in filename or 'hap' in filename:
            emotion = 'happy'
        elif 'neutral' in filename or 'neu' in filename:
            emotion = 'neutral'
        elif 'sad' in filename:
            emotion = 'sad'
        else:
            continue
        
        files.append(str(audio_file))
        labels.append(EMOTION_TO_INDEX[emotion])
    
    return files, labels

def map_model_labels_to_ours(model_labels):
    """Map model's label set to our standard 5 emotions"""
    # Common mappings
    mapping = {}
    
    for model_label in model_labels:
        ml_lower = model_label.lower()
        
        if 'ang' in ml_lower:
            mapping[model_label] = 'angry'
        elif 'fear' in ml_lower or 'fea' in ml_lower:
            mapping[model_label] = 'fear'
        elif 'hap' in ml_lower or 'joy' in ml_lower:
            mapping[model_label] = 'happy'
        elif 'neu' in ml_lower or 'calm' in ml_lower:
            mapping[model_label] = 'neutral'
        elif 'sad' in ml_lower:
            mapping[model_label] = 'sad'
        else:
            # Map unknown to neutral as default
            mapping[model_label] = 'neutral'
    
    return mapping

def test_model_on_dataset(model_name, files, labels, language, batch_size=8):
    """Test a pretrained model on a dataset"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Language: {language}")
    print('='*70)
    
    try:
        # Load model and processor
        print(f"\nLoading model from HuggingFace...")
        
        if 'hubert' in model_name.lower():
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = HubertForSequenceClassification.from_pretrained(model_name)
        else:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Get label mapping
        model_labels = list(model.config.id2label.values())
        label_mapping = map_model_labels_to_ours(model_labels)
        
        print(f"✓ Model loaded (device: {device})")
        print(f"\nModel's emotion labels: {model_labels}")
        print(f"Label mapping to our format: {label_mapping}")
        
        # Split into test set (use last 15% as test)
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(files))
        _, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
        
        test_files = [files[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        print(f"\nTest samples: {len(test_files)}")
        
        # Predict
        all_preds = []
        print(f"\nPredicting...")
        
        for i in tqdm(range(0, len(test_files), batch_size)):
            batch_files = test_files[i:i+batch_size]
            
            # Load audio batch
            batch_audios = []
            for file_path in batch_files:
                audio, sr = librosa.load(file_path, sr=TARGET_SR)
                batch_audios.append(audio)
            
            # Process
            inputs = processor(batch_audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Map predictions to our labels
            for pred in predictions.cpu().numpy():
                model_emotion = model.config.id2label[pred]
                our_emotion = label_mapping[model_emotion]
                all_preds.append(EMOTION_TO_INDEX[our_emotion])
        
        all_preds = np.array(all_preds)
        test_labels = np.array(test_labels)
        
        # Calculate accuracy
        accuracy = 100. * (all_preds == test_labels).sum() / len(test_labels)
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print('='*70)
        print(f"\nTest Accuracy: {accuracy:.2f}%\n")
        
        # Per-class accuracy
        print("Per-class test accuracy:")
        for i in range(5):
            mask = test_labels == i
            if mask.sum() > 0:
                class_acc = 100. * (all_preds[mask] == test_labels[mask]).sum() / mask.sum()
                count = mask.sum()
                print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({count:3d} samples)")
        
        return accuracy
        
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("TESTING EMOTION-SPECIFIC PRETRAINED MODELS")
    print("=" * 70)
    print("\nWe'll test models that were specifically fine-tuned for emotion")
    print("recognition, unlike general speech models like emotion2vec.\n")
    
    # Models to test
    models = [
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        "Rajaram1996/Hubert_emotion",
    ]
    
    # Collect data
    print("Collecting data...")
    english_files, english_labels = collect_english_files()
    tamil_files, tamil_labels = collect_tamil_files()
    
    print(f"✓ English: {len(english_files)} files")
    print(f"✓ Tamil: {len(tamil_files)} files")
    
    results = {}
    
    # Test each model
    for model_name in models:
        model_short_name = model_name.split('/')[-1]
        results[model_short_name] = {}
        
        print("\n" + "█" * 70)
        print(f"MODEL: {model_short_name}")
        print("█" * 70)
        
        # Test on English
        if len(english_files) > 0:
            acc = test_model_on_dataset(model_name, english_files, english_labels, "English", batch_size=8)
            results[model_short_name]['English'] = acc
        
        # Test on Tamil
        if len(tamil_files) > 0:
            acc = test_model_on_dataset(model_name, tamil_files, tamil_labels, "Tamil", batch_size=8)
            results[model_short_name]['Tamil'] = acc
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("ENGLISH RESULTS")
    print("-" * 70)
    print(f"{'Model':<50s} {'Accuracy':>10s}")
    print("-" * 70)
    print(f"{'CNN (baseline)':<50s} {'94.89%':>10s}")
    print(f"{'emotion2vec frozen':<50s} {'31.65%':>10s}")
    for model_name, res in results.items():
        if 'English' in res and res['English'] is not None:
            print(f"{model_name:<50s} {res['English']:>9.2f}%")
    
    print("\n" + "-" * 70)
    print("TAMIL RESULTS")
    print("-" * 70)
    print(f"{'Model':<50s} {'Accuracy':>10s}")
    print("-" * 70)
    print(f"{'Feature-based MLP (augmented)':<50s} {'36.88%':>10s}")
    print(f"{'Simple CNN':<50s} {'34.04%':>10s}")
    print(f"{'emotion2vec frozen':<50s} {'20.57%':>10s}")
    for model_name, res in results.items():
        if 'Tamil' in res and res['Tamil'] is not None:
            print(f"{model_name:<50s} {res['Tamil']:>9.2f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Find best pretrained model for English
    best_eng = max([(m, r['English']) for m, r in results.items() if 'English' in r and r['English'] is not None], 
                   key=lambda x: x[1] if x[1] else 0, default=(None, 0))
    
    if best_eng[1] and best_eng[1] > 90:
        print(f"\n✅ EXCELLENT! {best_eng[0]} achieves {best_eng[1]:.2f}% on English!")
        print(f"   This rivals the CNN baseline (94.89%).")
    elif best_eng[1] and best_eng[1] > 80:
        print(f"\n✓ Good! {best_eng[0]} achieves {best_eng[1]:.2f}% on English.")
        print(f"   Still below CNN (94.89%) but much better than emotion2vec.")
    elif best_eng[1] and best_eng[1] > 50:
        print(f"\n⚠️  {best_eng[0]} achieves {best_eng[1]:.2f}% on English.")
        print(f"   Better than emotion2vec but not production-ready.")
    else:
        print(f"\n❌ Emotion-specific models didn't perform well.")
        print(f"   Stick with CNN for English (94.89%).")
    
    print()

if __name__ == '__main__':
    main()
