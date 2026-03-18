"""
Test the CNN model with multiple audio samples
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
print("Loading model...")
model = keras.models.load_model(model_path)

# Test files
test_files = [
    ("03-01-03-01-01-01-01.wav", "happy"),
    ("03-01-04-01-01-01-01.wav", "sad"),
    ("03-01-05-01-01-01-01.wav", "angry"),
    ("03-01-06-01-01-01-01.wav", "fear"),
    ("03-01-07-01-01-01-01.wav", "disgust")
]

print("\n" + "="*80)
print("                    🎤 TESTING MULTIPLE AUDIO SAMPLES")
print("="*80 + "\n")

correct = 0
total = 0

for filename, expected_emotion in test_files:
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
        correct += 1
    total += 1
    
    # Display
    status = "✅" if is_correct else "❌"
    print(f"{status} {filename}")
    print(f"   Expected: {expected_emotion:8s} | Predicted: {predicted_emotion:8s} | Confidence: {confidence:.1%}")
    print()

print("="*80)
print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.1f}%")
print("="*80)
