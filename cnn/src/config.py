"""
Configuration file for Mel Spectrogram-based Speech Emotion Recognition
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_spectrograms')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs', 'training_logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset paths
RAVDESS_PATH = os.path.join(RAW_DATA_DIR, 'RAVDESS-SPEECH')
TESS_PATH = os.path.join(RAW_DATA_DIR, 'TESS', 'TESS Toronto emotional speech set data')
EMOTA_PATH = os.path.join(RAW_DATA_DIR, 'EMOTA', 'TamilSER-DB')
SINHALA_PATH = os.path.join(RAW_DATA_DIR, 'SINHALA')

# Audio processing parameters for mel spectrogram
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MELS = 128  # Height of spectrogram
N_FFT = 2048
HOP_LENGTH = 512
MAX_TIME_STEPS = 130  # Width of spectrogram (will pad/truncate to this)

# Model parameters
BATCH_SIZE = 16  # Reduced for stability
EPOCHS = 50  # Increased for proper training
LEARNING_RATE = 0.001  # Increased for better initial learning
VALIDATION_SPLIT = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Image augmentation
USE_AUGMENTATION = True

# RAVDESS emotion mapping
# RAVDESS filename: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
# Position 3 (index 2 when split by '-') is the emotion code
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprised'
}

# TESS emotions are in folder names (e.g., OAF_angry, YAF_happy)
TESS_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Target emotions for multilingual training (5 core emotions)
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']

# Emotion to index mapping
EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
INDEX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

# Create directories
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"Configuration loaded!")
print(f"Target emotions: {EMOTIONS}")
print(f"Mel spectrogram shape: ({N_MELS}, {MAX_TIME_STEPS})")