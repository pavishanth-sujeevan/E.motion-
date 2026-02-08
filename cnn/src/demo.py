"""
Demo script to test the complete SER pipeline
This script demonstrates how to use the preprocessing, training, and evaluation modules
"""
import os
import sys


def check_datasets():
    """Check if datasets are properly placed"""
    print("=" * 60)
    print("CHECKING DATASET AVAILABILITY")
    print("=" * 60)

    from config import RAVDESS_PATH, TESS_PATH

    ravdess_exists = os.path.exists(RAVDESS_PATH)
    tess_exists = os.path.exists(TESS_PATH)

    print(f"\nRAVDESS dataset: {'✓ Found' if ravdess_exists else '✗ Not found'}")
    print(f"  Path: {RAVDESS_PATH}")

    if ravdess_exists:
        actors = [f for f in os.listdir(RAVDESS_PATH) if f.startswith('Actor_')]
        print(f"  Actors found: {len(actors)}")

    print(f"\nTESS dataset: {'✓ Found' if tess_exists else '✗ Not found'}")
    print(f"  Path: {TESS_PATH}")

    if tess_exists:
        emotions = [f for f in os.listdir(TESS_PATH) if os.path.isdir(os.path.join(TESS_PATH, f))]
        print(f"  Emotion folders found: {len(emotions)}")
        print(f"  Emotions: {emotions}")

    if not ravdess_exists or not tess_exists:
        print("\n⚠ Warning: Please ensure datasets are placed in the correct directories!")
        print("\nExpected structure:")
        print("  data/raw/RAVDESS-SPEECH/Actor_XX/")
        print("  data/raw/TESS/TESS Toronto emotional speech set data/OAF_emotion/")
        return False

    return True


def check_processed_data():
    """Check if preprocessed data exists"""
    print("\n" + "=" * 60)
    print("CHECKING PREPROCESSED DATA")
    print("=" * 60)

    from config import PROCESSED_DATA_DIR

    required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy',
                      'label_classes.npy', 'feature_mean.npy', 'feature_std.npy']

    all_exist = True
    for filename in required_files:
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        exists = os.path.exists(filepath)
        print(f"  {filename:20s}: {'✓ Found' if exists else '✗ Not found'}")
        all_exist = all_exist and exists

    if all_exist:
        import numpy as np
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
        print(f"\n  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Feature dimension: {X_train.shape[1]}")

    return all_exist


def check_trained_model():
    """Check if trained model exists"""
    print("\n" + "=" * 60)
    print("CHECKING TRAINED MODEL")
    print("=" * 60)

    from config import MODELS_DIR

    model_path = os.path.join(MODELS_DIR, 'final_model.h5')
    exists = os.path.exists(model_path)

    print(f"  final_model.h5: {'✓ Found' if exists else '✗ Not found'}")

    if exists:
        import os
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")

    # Check for other saved models
    if os.path.exists(MODELS_DIR):
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
        if len(models) > 1:
            print(f"\n  Other saved models found: {len(models) - 1}")

    return exists


def run_demo():
    """Run a complete demo of the pipeline"""
    print("\n" + "=" * 60)
    print("SPEECH EMOTION RECOGNITION - DEMO")
    print("=" * 60)

    # Check datasets
    datasets_ok = check_datasets()

    # Check preprocessed data
    data_processed = check_processed_data()

    # Check trained model
    model_exists = check_trained_model()

    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    print(f"\n1. Datasets available: {'✓' if datasets_ok else '✗'}")
    print(f"2. Data preprocessed: {'✓' if data_processed else '✗'}")
    print(f"3. Model trained: {'✓' if model_exists else '✗'}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    if not datasets_ok:
        print("\n→ Place datasets in data/raw/ directory")
        print("  Then run: python preprocess.py")
    elif not data_processed:
        print("\n→ Run preprocessing: python preprocess.py")
    elif not model_exists:
        print("\n→ Train the model: python train.py")
    else:
        print("\n→ Pipeline is complete! You can:")
        print("  - Evaluate the model: python evaluate.py")
        print("  - Make predictions on new audio files")

        # Try to show a sample prediction
        try:
            print("\n" + "=" * 60)
            print("SAMPLE PREDICTION DEMO")
            print("=" * 60)

            from config import PROCESSED_DATA_DIR
            import numpy as np

            # Load a random sample from test set
            X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
            y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
            label_classes = np.load(os.path.join(PROCESSED_DATA_DIR, 'label_classes.npy'))

            # Make prediction on first test sample
            from tensorflow import keras
            from config import MODELS_DIR

            model = keras.models.load_model(os.path.join(MODELS_DIR, 'final_model.h5'))

            sample_idx = 0
            prediction = model.predict(X_test[sample_idx:sample_idx + 1], verbose=0)[0]
            predicted_class = np.argmax(prediction)
            true_class = y_test[sample_idx]

            print(f"\nSample Test Prediction:")
            print(f"  True emotion: {label_classes[true_class]}")
            print(f"  Predicted emotion: {label_classes[predicted_class]}")
            print(f"  Confidence: {prediction[predicted_class] * 100:.2f}%")
            print(f"  Match: {'✓ Correct' if predicted_class == true_class else '✗ Incorrect'}")

            print("\n  All probabilities:")
            for i, (emotion, prob) in enumerate(zip(label_classes, prediction)):
                marker = "→" if i == predicted_class else " "
                print(f"  {marker} {emotion:12s}: {prob * 100:5.2f}%")

        except Exception as e:
            print(f"\nCouldn't run sample prediction: {str(e)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_demo()