"""
Simple Emotion Predictor - No Command Line Required!

HOW TO USE:
1. Edit the AUDIO_FILE variable below with your audio file path
2. Run this file in PyCharm (Right-click -> Run)
3. See the results!
"""
import os
import sys
import numpy as np
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from preprocess import extract_features


# ============================================================
# EDIT THIS: Put your audio file path here
# ============================================================
AUDIO_FILE = "03-01-08-02-02-02-01.wav"
# ============================================================


def predict_emotion():
    """Predict emotion from the audio file"""

    print("\n" + "="*70)
    print(" " * 20 + "🎤 EMOTION PREDICTION")
    print("="*70)

    # Check if audio file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"\n❌ ERROR: Audio file not found!")
        print(f"   Looking for: {AUDIO_FILE}")
        print(f"\n💡 TIP: Edit the AUDIO_FILE variable in this script")
        print(f"        to point to your audio file.")
        return

    print(f"\n📁 Audio File: {AUDIO_FILE}")
    print(f"   Filename: {os.path.basename(AUDIO_FILE)}")

    # Load model
    model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found!")
        print(f"   Looking for: {model_path}")
        print(f"\n💡 TIP: Train the model first by running:")
        print(f"        python src/train.py")
        return

    print(f"\n🤖 Loading trained model...")
    model = keras.models.load_model(model_path)
    print(f"   ✓ Model loaded successfully!")

    # Load preprocessing parameters
    try:
        mean = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'feature_mean.npy'))
        std = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'feature_std.npy'))
        label_classes = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'))
        print(f"   ✓ Preprocessing parameters loaded!")
    except:
        print(f"\n❌ ERROR: Preprocessing files not found!")
        print(f"   Run preprocessing first: python src/preprocess.py")
        return

    # Extract features
    print(f"\n🎵 Extracting audio features...")
    features = extract_features(AUDIO_FILE)

    if features is None:
        print(f"   ❌ Failed to extract features!")
        print(f"   Make sure the file is a valid audio file (.wav recommended)")
        return

    print(f"   ✓ Features extracted! (Dimension: {len(features)})")

    # Normalize features
    features = (features - mean) / (std + 1e-8)
    features = features.reshape(1, -1)

    # Make prediction
    print(f"\n🔮 Making prediction...")
    predictions = model.predict(features, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    predicted_emotion = label_classes[predicted_class]
    confidence = predictions[predicted_class]

    # Display results with nice formatting
    print("\n" + "="*70)
    print(" " * 25 + "🎯 RESULTS")
    print("="*70)

    print(f"\n{'PREDICTED EMOTION:':>30} {predicted_emotion.upper()}")
    print(f"{'CONFIDENCE:':>30} {confidence:.2%}")

    # Create confidence bar
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
        emotion = label_classes[idx]
        prob = predictions[idx]

        # Create progress bar
        bar_length = 30
        filled = int(prob * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Marker for predicted emotion
        marker = "👉" if idx == predicted_class else "  "

        print(f"{marker} {rank}. {emotion.upper():12s} {bar} {prob:6.2%}")

    print("\n" + "="*70)
    print(" " * 22 + "✨ PREDICTION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    predict_emotion()