"""
Configuration file for Speech Emotion Recognition
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs', 'training_logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset paths
RAVDESS_PATH = os.path.join(RAW_DATA_DIR, 'RAVDESS-SPEECH')
TESS_PATH = os.path.join(RAW_DATA_DIR, 'TESS', 'TESS Toronto emotional speech set data')

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Model parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Emotion labels mapping
# RAVDESS emotions: 01 = neutral, 02 = calm, 03 = happy, 04 = sad,
#                   05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
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

# TESS emotions are in folder names
TESS_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Combined emotions (intersection of both datasets for consistency)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad']

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)