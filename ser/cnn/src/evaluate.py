"""
Evaluation script for Mel Spectrogram CNN model
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras

import config


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})

    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_model(model_path=None):
    """Evaluate the trained model"""
    print("="*70)
    print("EVALUATING MEL SPECTROGRAM CNN")
    print("="*70)

    # Load model
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'spectrogram_model_final.h5')

    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Load test data
    print("Loading test data...")
    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))

    print(f"\nTest set: {X_test.shape[0]} samples")
    print(f"Classes: {config.EMOTIONS}")

    # Predict
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*70}")
    report = classification_report(
        y_test, y_pred,
        target_names=config.EMOTIONS,
        digits=4
    )
    print(report)

    # Per-class accuracy
    print(f"{'='*70}")
    print("PER-CLASS DETAILED ACCURACY")
    print(f"{'='*70}")
    for idx in range(len(config.EMOTIONS)):
        emotion = config.INDEX_TO_EMOTION[idx]
        mask = y_test == idx
        if mask.sum() > 0:
            class_correct = (y_pred[mask] == idx).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total

            # Get average confidence for correct predictions
            correct_mask = (y_test == idx) & (y_pred == idx)
            if correct_mask.sum() > 0:
                avg_confidence = y_pred_probs[correct_mask, idx].mean()
            else:
                avg_confidence = 0.0

            print(f"{emotion:12s}: Accuracy {class_acc:.4f} ({class_acc*100:5.1f}%) | "
                  f"Correct {class_correct}/{class_total} | "
                  f"Avg Confidence {avg_confidence:.4f}")

    # Save confusion matrix
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix_spectrogram.png')
    plot_confusion_matrix(cm, config.EMOTIONS, cm_path)

    # Save report
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report_spectrogram.txt')
    with open(report_path, 'w') as f:
        f.write("MEL SPECTROGRAM CNN - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}\n")

    return accuracy, cm


if __name__ == "__main__":
    evaluate_model()