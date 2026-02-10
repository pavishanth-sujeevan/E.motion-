"""
Simple prediction script for Mel Spectrogram CNN
No command line - just edit the audio file path and run!
"""
import os
import sys
import numpy as np
import librosa
from tensorflow import keras

import config_spectrogram as config


# ============================================================
# EDIT THIS: Put your audio file path here
# ============================================================
AUDIO_FILE = "data/raw/RAVDESS-SPEECH/Actor_01/03-01-01-01-01-01-01.wav"
# ============================================================


def extract_melspectrogram(file_path):
    """Extract mel spectrogram from audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION)

        # Ensure fixed length
        max_len = config.SAMPLE_RATE * config.DURATION
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]

        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # Ensure fixed time dimension
        if mel_spec_normalized.shape[1] < config.MAX_TIME_STEPS:
            pad_width = config.MAX_TIME_STEPS - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :config.MAX_TIME_STEPS]

        return mel_spec_normalized

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def predict_emotion():
    """Predict emotion from audio file"""

    print("\n" + "="*70)
    print(" " * 15 + "🎤 MEL SPECTROGRAM EMOTION PREDICTION")
    print("="*70)

    # Check file
    if not os.path.exists(AUDIO_FILE):
        print(f"\n❌ ERROR: Audio file not found!")
        print(f"   Looking for: {AUDIO_FILE}")
        print(f"\n💡 TIP: Edit the AUDIO_FILE variable at the top of this script")
        return

    print(f"\n📁 Audio File: {AUDIO_FILE}")
    print(f"   Filename: {os.path.basename(AUDIO_FILE)}")

    # Load model
    model_path = os.path.join(config.MODELS_DIR, 'spectrogram_model_final.h5')

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found!")
        print(f"   Looking for: {model_path}")
        print(f"\n💡 TIP: Train the model first:")
        print(f"        python train_spectrogram.py")
        return

    print(f"\n🤖 Loading trained model...")
    model = keras.models.load_model(model_path)
    print(f"   ✓ Model loaded successfully!")

    # Extract mel spectrogram
    print(f"\n🎵 Extracting mel spectrogram...")
    mel_spec = extract_melspectrogram(AUDIO_FILE)

    if mel_spec is None:
        print(f"   ❌ Failed to extract mel spectrogram!")
        return

    print(f"   ✓ Mel spectrogram extracted!")
    print(f"   Shape: {mel_spec.shape} (n_mels × time_steps)")

    # Prepare for prediction
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    # Predict
    print(f"\n🔮 Making prediction...")
    predictions = model.predict(mel_spec, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    predicted_emotion = config.INDEX_TO_EMOTION[predicted_class]
    confidence = predictions[predicted_class]

    # Display results
    print("\n" + "="*70)
    print(" " * 25 + "🎯 RESULTS")
    print("="*70)

    print(f"\n{'PREDICTED EMOTION:':>30} {predicted_emotion.upper()}")
    print(f"{'CONFIDENCE:':>30} {confidence:.2%}")

    # Confidence bar
    bar_length = 40
    filled = int(confidence * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"{'':>30} [{bar}]")

    print("\n" + "-"*70)
    print(" " * 20 + "📊 ALL EMOTION PROBABILITIES")
    print("-"*70)

    # Sort by probability
    sorted_indices = np.argsort(predictions)[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        emotion = config.INDEX_TO_EMOTION[idx]
        prob = predictions[idx]

        # Progress bar
        bar_length = 30
        filled = int(prob * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Marker
        marker = "👉" if idx == predicted_class else "  "

        print(f"{marker} {rank}. {emotion.upper():12s} {bar} {prob:6.2%}")

    # Interpretation
    print("\n" + "="*70)
    print(" " * 20 + "📝 CONFIDENCE INTERPRETATION")
    print("="*70)

    if confidence > 0.8:
        interpretation = "Very High - Model is very confident"
    elif confidence > 0.6:
        interpretation = "High - Model is confident"
    elif confidence > 0.4:
        interpretation = "Moderate - Model has some uncertainty"
    else:
        interpretation = "Low - Model is uncertain, check the audio quality"

    print(f"\n{interpretation}")

    print("\n" + "="*70)
    print(" " * 22 + "✨ PREDICTION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    predict_emotion()