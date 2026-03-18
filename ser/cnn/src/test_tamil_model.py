"""
Test the Tamil-only model with Tamil audio samples.
"""
import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import config
from src import config

def load_and_preprocess_audio(file_path):
    """Load and preprocess audio file."""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
        
        # Pad or trim
        if len(y) < config.SAMPLE_RATE * config.DURATION:
            y = np.pad(y, (0, config.SAMPLE_RATE * config.DURATION - len(y)))
        else:
            y = y[:config.SAMPLE_RATE * config.DURATION]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        # Pad or trim time dimension
        if mel_spec_db.shape[1] < config.MAX_TIME_STEPS:
            pad_width = config.MAX_TIME_STEPS - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :config.MAX_TIME_STEPS]
        
        # Add channel dimension
        mel_spec_db = mel_spec_db[..., np.newaxis]
        
        return mel_spec_db
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_tamil_samples():
    """Get some Tamil sample files from EMOTA dataset."""
    emota_path = config.EMOTA_PATH
    samples = []
    
    # Mapping from file abbreviations to emotion labels
    emotion_mapping = {
        'ang': 'angry',
        'fea': 'fear',
        'hap': 'happy',
        'neu': 'neutral',
        'sad': 'sad'
    }
    
    # Get 3 samples from each emotion
    emotion_counts = {label: 0 for label in emotion_mapping.values()}
    max_per_emotion = 3
    
    for file in sorted(os.listdir(emota_path)):
        if not file.endswith('.wav'):
            continue
        
        # Extract emotion from filename (format: XX_XX_EMOTION.wav)
        parts = file.split('_')
        if len(parts) >= 3:
            emotion_abbr = parts[2].split('.')[0]
            
            if emotion_abbr in emotion_mapping:
                emotion_label = emotion_mapping[emotion_abbr]
                
                # Only add if we haven't reached max for this emotion
                if emotion_counts[emotion_label] < max_per_emotion:
                    samples.append((os.path.join(emota_path, file), emotion_label))
                    emotion_counts[emotion_label] += 1
    
    return samples

def test_tamil_model():
    """Test the Tamil model."""
    print("=" * 80)
    print("Testing Tamil Simple Model (Best Performing)")
    print("=" * 80)
    
    # Load model
    model_path = "E:\\Projects\\E.motion-\\cnn\\models\\saved_models\\language_models\\tamil\\tamil_simple_model.h5"
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[OK] Model loaded successfully")
    
    # Get Tamil samples
    print("\nGetting Tamil samples from EMOTA dataset...")
    samples = get_tamil_samples()
    
    if not samples:
        print("ERROR: No Tamil samples found!")
        return
    
    print(f"Found {len(samples)} Tamil samples")
    
    # Test each sample
    correct = 0
    total = 0
    confidences = []
    
    print("\n" + "=" * 80)
    print("Testing Samples")
    print("=" * 80)
    
    for file_path, true_label in samples:
        # Preprocess
        mel_spec = load_and_preprocess_audio(file_path)
        
        if mel_spec is None:
            continue
        
        # Predict
        mel_spec_batch = np.expand_dims(mel_spec, axis=0)
        predictions = model.predict(mel_spec_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_label = config.EMOTIONS[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
        
        # Check if correct
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
        total += 1
        confidences.append(confidence)
        
        # Display result
        status = "[CORRECT]" if is_correct else "[WRONG]"
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"True: {true_label:8s} | Predicted: {predicted_label:8s} | Confidence: {confidence:5.1f}% {status}")
        
        # Show all predictions
        print("All predictions:")
        for i, emotion in enumerate(config.EMOTIONS):
            prob = predictions[0][i] * 100
            bar = "#" * int(prob / 2)
            print(f"  {emotion:8s}: {prob:5.1f}% {bar}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    test_tamil_model()
