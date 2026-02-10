"""
IMPROVED Training script with better model architecture and class weighting
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config


def build_improved_cnn_model(input_shape, num_classes):
    """
    Build improved CNN model with better architecture
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_shape,)),

        # Reshape for Conv1D
        layers.Reshape((input_shape, 1)),

        # First Conv Block
        layers.Conv1D(128, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv1D(128, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Second Conv Block
        layers.Conv1D(256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv1D(256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Third Conv Block
        layers.Conv1D(512, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),

        # Global pooling
        layers.GlobalAveragePooling1D(),

        # Dense layers
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def train_model():
    """Main training function with improved settings"""
    print("="*60)
    print("STARTING IMPROVED TRAINING")
    print("="*60)

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    label_classes = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'))

    num_classes = len(label_classes)

    print(f"\n{'='*60}")
    print("DATASET INFORMATION")
    print(f"{'='*60}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_classes}")

    # Check class distribution
    print(f"\nTraining set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for idx, count in zip(unique, counts):
        emotion = label_classes[idx]
        percentage = (count / len(y_train)) * 100
        print(f"  {emotion:12s}: {count:4d} ({percentage:5.1f}%)")

    # Compute class weights to handle any remaining imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\nClass weights:")
    for idx, weight in class_weight_dict.items():
        emotion = label_classes[idx]
        print(f"  {emotion:12s}: {weight:.4f}")

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )

    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    print(f"\n{'='*60}")
    print("AFTER VALIDATION SPLIT")
    print(f"{'='*60}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Build model
    print(f"\n{'='*60}")
    print("BUILDING IMPROVED CNN MODEL")
    print(f"{'='*60}")
    model = build_improved_cnn_model(X_train.shape[1], num_classes)

    # Compile model with label smoothing
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.MODELS_DIR, f'best_model_{timestamp}.h5')

    callback_list = [
        # Save best model
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # Early stopping with more patience
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard
        callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f'logs_{timestamp}'),
            histogram_freq=1
        )
    ]

    # Train model
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Using class weights: Yes")
    print(f"Early stopping patience: 20")
    print(f"{'='*60}\n")

    history = model.fit(
        X_train, y_train_cat,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callback_list,
        class_weight=class_weight_dict,  # Use class weights
        verbose=1
    )

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*60}")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Get per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\nPer-class test accuracy:")
    for idx in range(num_classes):
        emotion = label_classes[idx]
        mask = y_test == idx
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == idx).sum() / mask.sum()
            print(f"  {emotion:12s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Plot training history
    history_plot_path = os.path.join(config.RESULTS_DIR, f'training_history_{timestamp}.png')
    plot_training_history(history, history_plot_path)

    # Save training configuration
    training_config = {
        'timestamp': timestamp,
        'model_path': model_path,
        'num_classes': num_classes,
        'classes': label_classes.tolist(),
        'feature_dim': int(X_train.shape[1]),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'epochs_trained': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'batch_size': config.BATCH_SIZE,
        'learning_rate': 0.0005,
        'used_class_weights': True,
        'class_weights': {label_classes[k]: float(v) for k, v in class_weight_dict.items()}
    }

    config_path = os.path.join(config.RESULTS_DIR, f'training_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)

    print(f"Training configuration saved to: {config_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nBest model: {model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"\nNext step: Evaluate the model")
    print(f"  python src/evaluate.py")
    print(f"{'='*60}\n")

    return model, history


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(config.RANDOM_STATE)
    tf.random.set_seed(config.RANDOM_STATE)

    # Train model
    model, history = train_model()