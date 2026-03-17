"""
Test emotion models using pipeline API (simpler approach)
Try models that work with standard transformers pipelines
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
warnings.filterwarnings('ignore')

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}
TARGET_SR = 16000

def collect_test_files(language):
    """Collect test files for a language"""
    files = []
    labels = []
    
    if language == 'English':
        data_root = Path('../data/raw')
        
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
    
    elif language == 'Tamil':
        data_root = Path('../data/raw/EmoTa/TamilSER-DB')
        if data_root.exists():
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
    
    # Use last 15% as test set
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(files))
    _, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    
    test_files = [files[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return test_files, test_labels

def map_prediction_to_standard(pred_label):
    """Map model prediction to our 5 standard emotions"""
    pl = pred_label.lower()
    
    if 'ang' in pl or 'anger' in pl:
        return 'angry'
    elif 'fear' in pl:
        return 'fear'
    elif 'hap' in pl or 'joy' in pl or 'surprise' in pl:
        return 'happy'
    elif 'neu' in pl or 'calm' in pl:
        return 'neutral'
    elif 'sad' in pl or 'disgust' in pl:
        return 'sad'
    else:
        # Default unknown to neutral
        return 'neutral'

def test_model_with_pipeline(model_name, test_files, test_labels, language):
    """Test model using pipeline API"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Language: {language} ({len(test_files)} test samples)")
    print('='*70)
    
    try:
        # Try to load with pipeline
        print(f"\nLoading model...")
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            classifier = pipeline("audio-classification", model=model_name, device=device)
            print(f"Loaded with pipeline API")
        except:
            # Try manual loading
            print("  Pipeline failed, trying manual load...")
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            model.eval()
            print(f"Loaded manually")
            
            # Create manual classifier
            def manual_classify(audio_file):
                audio, sr = librosa.load(audio_file, sr=TARGET_SR)
                inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
                    predicted_label = model.config.id2label[predicted_id]
                return [{'label': predicted_label, 'score': 1.0}]
            
            classifier = manual_classify
        
        # Get sample prediction to see labels
        sample_audio, _ = librosa.load(test_files[0], sr=TARGET_SR)
        if callable(classifier):
            sample_pred = classifier(test_files[0])
        else:
            sample_pred = classifier(sample_audio, sampling_rate=TARGET_SR)
        
        print(f"\nModel emotion labels: {sample_pred[0]['label']}")
        
        # Predict on all test files
        all_preds = []
        print(f"\nPredicting...")
        
        for audio_file in tqdm(test_files[:100]):  # Test on first 100 for speed
            try:
                if callable(classifier):
                    result = classifier(audio_file)
                else:
                    audio, sr = librosa.load(audio_file, sr=TARGET_SR)
                    result = classifier(audio, sampling_rate=TARGET_SR)
                
                # Get top prediction
                top_pred = result[0]['label']
                mapped_emotion = map_prediction_to_standard(top_pred)
                all_preds.append(EMOTION_TO_INDEX[mapped_emotion])
            except Exception as e:
                print(f"Error on file: {e}")
                all_preds.append(3)  # Default to neutral
        
        all_preds = np.array(all_preds)
        test_labels_subset = np.array(test_labels[:100])
        
        # Calculate accuracy
        accuracy = 100. * (all_preds == test_labels_subset).sum() / len(test_labels_subset)
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print('='*70)
        print(f"\nTest Accuracy: {accuracy:.2f}% (on 100 samples)\n")
        
        # Per-class accuracy
        print("Per-class accuracy:")
        for i in range(5):
            mask = test_labels_subset == i
            if mask.sum() > 0:
                class_acc = 100. * (all_preds[mask] == test_labels_subset[mask]).sum() / mask.sum()
                count = mask.sum()
                print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({count:3d} samples)")
        
        return accuracy
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("TESTING EMOTION MODELS WITH PIPELINE API")
    print("=" * 70)
    print()
    
    # Try simpler models that work with standard pipelines
    models_to_try = [
        "superb/wav2vec2-base-superb-er",  # Standard emotion recognition
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "Rajaram1996/Hubert_emotion",
    ]
    
    # Collect data
    print("Collecting test data...")
    english_files, english_labels = collect_test_files('English')
    tamil_files, tamil_labels = collect_test_files('Tamil')
    
    print(f"English: {len(english_files)} test files")
    print(f"Tamil: {len(tamil_files)} test files")
    
    results = {}
    
    for model_name in models_to_try:
        model_short = model_name.split('/')[-1]
        results[model_short] = {}
        
        print("\n" + "█" * 70)
        print(f"MODEL: {model_short}")
        print("█" * 70)
        
        # Test English
        if len(english_files) > 0:
            acc = test_model_with_pipeline(model_name, english_files, english_labels, "English")
            results[model_short]['English'] = acc
        
        # Test Tamil
        if len(tamil_files) > 0:
            acc = test_model_with_pipeline(model_name, tamil_files, tamil_labels, "Tamil")
            results[model_short]['Tamil'] = acc
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nEnglish:")
    print(f"  CNN baseline: 94.89%")
    for model, res in results.items():
        if 'English' in res and res['English']:
            print(f"  {model}: {res['English']:.2f}%")
    
    print("\nTamil:")
    print(f"  Feature-based MLP: 36.88%")
    for model, res in results.items():
        if 'Tamil' in res and res['Tamil']:
            print(f"  {model}: {res['Tamil']:.2f}%")

if __name__ == '__main__':
    main()
