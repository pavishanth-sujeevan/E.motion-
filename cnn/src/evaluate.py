"""
Evaluation script for Speech Emotion Recognition model
Generates confusion matrix, classification report, and other metrics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras

import config


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})

    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

    # Also plot raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    save_path_counts = save_path.replace('.png', '_counts.png')
    plt.savefig(save_path_counts, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix (counts) saved to: {save_path_counts}")
    plt.close()


def plot_per_class_metrics(precision, recall, f1, class_names, save_path):
    """
    Plot per-class precision, recall, and F1-score

    Args:
        precision: Precision scores for each class
        recall: Recall scores for each class
        f1: F1-scores for each class
        class_names: List of class names
        save_path: Path to save the plot
    """
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')

    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to: {save_path}")
    plt.close()


def evaluate_model(model_path=None):
    """
    Evaluate trained model on test set

    Args:
        model_path: Path to saved model (if None, uses final_model.h5)
    """
    print("Starting evaluation...")

    # Load model
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Load test data
    print("Loading test data...")
    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    label_classes = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'))

    print(f"\nTest set size: {X_test.shape[0]} samples")
    print(f"Number of classes: {len(label_classes)}")
    print(f"Classes: {label_classes}")

    # Make predictions
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    print("\nCalculating metrics...")

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_classes))
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(label_classes)))

    # Print detailed classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=label_classes,
        digits=4
    )
    print(report)

    # Print per-class accuracy
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)
    for i, emotion in enumerate(label_classes):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            print(f"{emotion:12s}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {support[i]} samples")

    # Macro and weighted averages
    print("\n" + "=" * 60)
    print("AVERAGE METRICS")
    print("=" * 60)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    print(f"Macro Average:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")

    print(f"\nWeighted Average:")
    print(f"  Precision: {weighted_precision:.4f}")
    print(f"  Recall:    {weighted_recall:.4f}")
    print(f"  F1-Score:  {weighted_f1:.4f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save confusion matrix plot
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_classes, cm_path)

    # Save per-class metrics plot
    metrics_path = os.path.join(config.RESULTS_DIR, 'per_class_metrics.png')
    plot_per_class_metrics(precision, recall, f1, label_classes, metrics_path)

    # Save classification report to file
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("SPEECH EMOTION RECOGNITION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("=" * 60 + "\n")
        for i, emotion in enumerate(label_classes):
            class_mask = y_test == i
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
                f.write(f"{emotion:12s}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {support[i]} samples\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 60 + "\n")
        f.write(np.array2string(cm, separator=', '))

    print(f"Classification report saved to: {report_path}")

    # Save predictions
    predictions_path = os.path.join(config.RESULTS_DIR, 'predictions.npz')
    np.savez(predictions_path,
             y_true=y_test,
             y_pred=y_pred,
             y_pred_probs=y_pred_probs,
             class_names=label_classes)
    print(f"Predictions saved to: {predictions_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def predict_emotion(audio_path, model_path=None):
    """
    Predict emotion from a single audio file

    Args:
        audio_path: Path to audio file
        model_path: Path to saved model (if None, uses final_model.h5)

    Returns:
        Predicted emotion and probability distribution
    """
    from preprocess import extract_features

    # Load model
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')

    model = keras.models.load_model(model_path)

    # Load normalization parameters and label classes
    mean = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'feature_mean.npy'))
    std = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'feature_std.npy'))
    label_classes = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'))

    # Extract features
    features = extract_features(audio_path)
    if features is None:
        print("Error extracting features from audio file")
        return None, None

    # Normalize
    features = (features - mean) / (std + 1e-8)

    # Reshape for model input
    features = features.reshape(1, -1)

    # Predict
    predictions = model.predict(features, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    predicted_emotion = label_classes[predicted_class]
    confidence = predictions[predicted_class]

    print(f"\nPredicted Emotion: {predicted_emotion}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
    print("\nAll probabilities:")
    for emotion, prob in zip(label_classes, predictions):
        print(f"  {emotion:12s}: {prob:.4f} ({prob * 100:.2f}%)")

    return predicted_emotion, predictions


if __name__ == "__main__":
    # Evaluate the model
    results = evaluate_model()