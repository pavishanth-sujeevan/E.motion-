"""
Test bilingual CNN model with English and Tamil samples
"""
import os
import numpy as np
import librosa
from tensorflow import keras
import config

def extract_melspectrogram(file_path):
    """Extract mel spectrogram from audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
        max_len = config.SAMPLE_RATE * config.DURATION
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=config.N_MELS,
            n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        if mel_spec_normalized.shape[1] < config.MAX_TIME_STEPS:
            pad_width = config.MAX_TIME_STEPS - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :config.MAX_TIME_STEPS]
        
        return mel_spec_normalized
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Load model
model_path = os.path.join(config.MODELS_DIR, 'spectrogram_model_final.h5')
print("="*80)
print("🌍 BILINGUAL EMOTION RECOGNITION TEST (English + Tamil)")
print("="*80)
print("\nLoading bilingual model...")
model = keras.models.load_model(model_path)
print("✓ Model loaded successfully!\n")

# English test files (RAVDESS)
english_tests = [
    ("03-01-05-02-01-01-01.wav", "angry", "English-RAVDESS"),
    ("03-01-06-02-01-01-01.wav", "fear", "English-RAVDESS"),
    ("03-01-03-02-01-01-01.wav", "happy", "English-RAVDESS"),
    ("03-01-01-02-01-01-01.wav", "neutral", "English-RAVDESS"),
    ("03-01-04-02-01-01-01.wav", "sad", "English-RAVDESS"),
]

# Tamil test files (EMOTA)
tamil_tests = [
    ("01_01_ang.wav", "angry", "Tamil-EMOTA"),
    ("01_01_fea.wav", "fear", "Tamil-EMOTA"),
    ("01_01_hap.wav", "happy", "Tamil-EMOTA"),
    ("01_01_neu.wav", "neutral", "Tamil-EMOTA"),
    ("01_01_sad.wav", "sad", "Tamil-EMOTA"),
]

all_tests = []

# Test English samples
print("="*80)
print("📢 TESTING ENGLISH SAMPLES")
print("="*80 + "\n")

english_correct = 0
english_total = 0

for filename, expected_emotion, dataset in english_tests:
    file_path = os.path.join(r"E:\Projects\E.motion-\cnn\data\raw\RAVDESS-SPEECH\Actor_01", filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️  {filename}: File not found")
        continue
    
    # Extract and predict
    mel_spec = extract_melspectrogram(file_path)
    if mel_spec is None:
        continue
    
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
    predictions = model.predict(mel_spec, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    predicted_emotion = config.INDEX_TO_EMOTION[predicted_class]
    confidence = predictions[predicted_class]
    
    # Check if correct
    is_correct = predicted_emotion == expected_emotion
    if is_correct:
        english_correct += 1
    english_total += 1
    
    # Display
    status = "✅" if is_correct else "❌"
    print(f"{status} {filename}")
    print(f"   Language: {dataset}")
    print(f"   Expected: {expected_emotion:8s} | Predicted: {predicted_emotion:8s} | Confidence: {confidence:.1%}")
    
    # Show top 3 predictions
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    print(f"   Top 3: ", end="")
    for idx in top_3_idx:
        emo = config.INDEX_TO_EMOTION[idx]
        prob = predictions[idx]
        print(f"{emo}({prob:.1%}) ", end="")
    print("\n")
    
    all_tests.append((dataset, expected_emotion, predicted_emotion, confidence, is_correct))

# Test Tamil samples
print("="*80)
print("🗣️  TESTING TAMIL SAMPLES")
print("="*80 + "\n")

tamil_correct = 0
tamil_total = 0

for filename, expected_emotion, dataset in tamil_tests:
    file_path = os.path.join(r"E:\Projects\E.motion-\cnn\data\raw\EMOTA\TamilSER-DB", filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️  {filename}: File not found")
        continue
    
    # Extract and predict
    mel_spec = extract_melspectrogram(file_path)
    if mel_spec is None:
        continue
    
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
    predictions = model.predict(mel_spec, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    predicted_emotion = config.INDEX_TO_EMOTION[predicted_class]
    confidence = predictions[predicted_class]
    
    # Check if correct
    is_correct = predicted_emotion == expected_emotion
    if is_correct:
        tamil_correct += 1
    tamil_total += 1
    
    # Display
    status = "✅" if is_correct else "❌"
    print(f"{status} {filename}")
    print(f"   Language: {dataset}")
    print(f"   Expected: {expected_emotion:8s} | Predicted: {predicted_emotion:8s} | Confidence: {confidence:.1%}")
    
    # Show top 3 predictions
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    print(f"   Top 3: ", end="")
    for idx in top_3_idx:
        emo = config.INDEX_TO_EMOTION[idx]
        prob = predictions[idx]
        print(f"{emo}({prob:.1%}) ", end="")
    print("\n")
    
    all_tests.append((dataset, expected_emotion, predicted_emotion, confidence, is_correct))

# Summary
print("="*80)
print("📊 SUMMARY")
print("="*80 + "\n")

print(f"English Samples:")
print(f"  Accuracy: {english_correct}/{english_total} = {(english_correct/english_total)*100:.1f}%")
print()

print(f"Tamil Samples:")
print(f"  Accuracy: {tamil_correct}/{tamil_total} = {(tamil_correct/tamil_total)*100:.1f}%")
print()

total_correct = english_correct + tamil_correct
total_samples = english_total + tamil_total
print(f"Overall:")
print(f"  Accuracy: {total_correct}/{total_samples} = {(total_correct/total_samples)*100:.1f}%")
print()

# Per-emotion accuracy
print("="*80)
print("📈 PER-EMOTION PERFORMANCE")
print("="*80 + "\n")

emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']
for emotion in emotions:
    emotion_tests = [t for t in all_tests if t[1] == emotion]
    if emotion_tests:
        emotion_correct = sum(1 for t in emotion_tests if t[4])
        emotion_total = len(emotion_tests)
        avg_confidence = np.mean([t[3] for t in emotion_tests])
        print(f"{emotion:8s}: {emotion_correct}/{emotion_total} ({(emotion_correct/emotion_total)*100:5.1f}%) - Avg confidence: {avg_confidence:.1%}")

print("\n" + "="*80)
print("✨ TESTING COMPLETE!")
print("="*80)
