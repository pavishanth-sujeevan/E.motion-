"""
Extended bilingual test with more samples
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
        return None

# Load model
model_path = os.path.join(config.MODELS_DIR, 'spectrogram_model_final.h5')
print("="*90)
print("🌍 EXTENDED BILINGUAL TEST - Multiple Actors/Speakers")
print("="*90)
print("\nLoading model...")
model = keras.models.load_model(model_path)
print("✓ Loaded!\n")

# Test multiple English actors
print("="*90)
print("📢 ENGLISH SAMPLES (RAVDESS - Multiple Actors)")
print("="*90 + "\n")

english_results = []
ravdess_base = r"E:\Projects\E.motion-\cnn\data\raw\RAVDESS-SPEECH"

# Emotion codes: 01=neutral, 03=happy, 04=sad, 05=angry, 06=fear
emotion_map = {'05': 'angry', '06': 'fear', '03': 'happy', '01': 'neutral', '04': 'sad'}

# Test first 3 actors with intensity 02
for actor in ['Actor_01', 'Actor_02', 'Actor_03']:
    actor_path = os.path.join(ravdess_base, actor)
    if not os.path.exists(actor_path):
        continue
    
    for emotion_code, emotion in emotion_map.items():
        # Get intensity 02, statement 01, repetition 01
        intensity = '02' if emotion != 'neutral' else '01'
        filename = f"03-01-{emotion_code}-{intensity}-01-01-{actor.split('_')[1]}.wav"
        file_path = os.path.join(actor_path, filename)
        
        if not os.path.exists(file_path):
            continue
        
        mel_spec = extract_melspectrogram(file_path)
        if mel_spec is None:
            continue
        
        mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
        predictions = model.predict(mel_spec, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        predicted_emotion = config.INDEX_TO_EMOTION[predicted_class]
        confidence = predictions[predicted_class]
        
        is_correct = predicted_emotion == emotion
        status = "✅" if is_correct else "❌"
        
        print(f"{status} {actor:12s} | Expected: {emotion:8s} | Predicted: {predicted_emotion:8s} | Conf: {confidence:.1%}")
        english_results.append((emotion, predicted_emotion, confidence, is_correct))

# Test Tamil samples (multiple speakers)
print("\n" + "="*90)
print("🗣️  TAMIL SAMPLES (EMOTA - Multiple Speakers)")
print("="*90 + "\n")

tamil_results = []
emota_base = r"E:\Projects\E.motion-\cnn\data\raw\EMOTA\TamilSER-DB"

# Test multiple speakers
emotion_suffixes = {'ang': 'angry', 'fea': 'fear', 'hap': 'happy', 'neu': 'neutral', 'sad': 'sad'}

for speaker in ['01', '02', '03', '04', '05']:
    for emotion_suffix, emotion in emotion_suffixes.items():
        filename = f"{speaker}_01_{emotion_suffix}.wav"
        file_path = os.path.join(emota_base, filename)
        
        if not os.path.exists(file_path):
            continue
        
        mel_spec = extract_melspectrogram(file_path)
        if mel_spec is None:
            continue
        
        mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
        predictions = model.predict(mel_spec, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        predicted_emotion = config.INDEX_TO_EMOTION[predicted_class]
        confidence = predictions[predicted_class]
        
        is_correct = predicted_emotion == emotion
        status = "✅" if is_correct else "❌"
        
        print(f"{status} Speaker {speaker} | Expected: {emotion:8s} | Predicted: {predicted_emotion:8s} | Conf: {confidence:.1%}")
        tamil_results.append((emotion, predicted_emotion, confidence, is_correct))

# Summary statistics
print("\n" + "="*90)
print("📊 DETAILED SUMMARY")
print("="*90 + "\n")

# English stats
eng_correct = sum(1 for r in english_results if r[3])
eng_total = len(english_results)
print(f"🇬🇧 English Performance:")
print(f"   Overall: {eng_correct}/{eng_total} = {(eng_correct/eng_total)*100:.1f}%")
print(f"   Avg Confidence: {np.mean([r[2] for r in english_results]):.1%}\n")

# English per-emotion
print("   Per-emotion breakdown:")
for emotion in ['angry', 'fear', 'happy', 'neutral', 'sad']:
    emo_results = [r for r in english_results if r[0] == emotion]
    if emo_results:
        correct = sum(1 for r in emo_results if r[3])
        total = len(emo_results)
        avg_conf = np.mean([r[2] for r in emo_results])
        print(f"      {emotion:8s}: {correct}/{total} ({(correct/total)*100:5.1f}%) - Avg conf: {avg_conf:.1%}")

# Tamil stats
print(f"\n🇱🇰 Tamil Performance:")
tam_correct = sum(1 for r in tamil_results if r[3])
tam_total = len(tamil_results)
print(f"   Overall: {tam_correct}/{tam_total} = {(tam_correct/tam_total)*100:.1f}%")
print(f"   Avg Confidence: {np.mean([r[2] for r in tamil_results]):.1%}\n")

# Tamil per-emotion
print("   Per-emotion breakdown:")
for emotion in ['angry', 'fear', 'happy', 'neutral', 'sad']:
    emo_results = [r for r in tamil_results if r[0] == emotion]
    if emo_results:
        correct = sum(1 for r in emo_results if r[3])
        total = len(emo_results)
        avg_conf = np.mean([r[2] for r in emo_results])
        print(f"      {emotion:8s}: {correct}/{total} ({(correct/total)*100:5.1f}%) - Avg conf: {avg_conf:.1%}")

# Overall stats
print(f"\n🌍 Combined Performance:")
total_correct = eng_correct + tam_correct
total_samples = eng_total + tam_total
print(f"   Overall: {total_correct}/{total_samples} = {(total_correct/total_samples)*100:.1f}%")

# Confusion analysis for Tamil
print("\n" + "="*90)
print("🔍 TAMIL CONFUSION ANALYSIS")
print("="*90 + "\n")

for expected_emotion in ['angry', 'fear', 'happy', 'neutral', 'sad']:
    emo_results = [r for r in tamil_results if r[0] == expected_emotion]
    if emo_results:
        predictions = {}
        for r in emo_results:
            pred = r[1]
            predictions[pred] = predictions.get(pred, 0) + 1
        
        print(f"{expected_emotion:8s} -> ", end="")
        for pred, count in sorted(predictions.items(), key=lambda x: -x[1]):
            print(f"{pred}:{count} ", end="")
        print()

print("\n" + "="*90)
print("✨ EXTENDED TESTING COMPLETE!")
print("="*90)
