"""
Training script for Speech Emotion Recognition CNN
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

import config


def build_cnn_model(input_shape, num_classes):
    """
    Build CNN model for emotion recognition

    Args:
        input_shape: Shape of input features
        num_classes: Number of emotion classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Reshape for CNN input (add channel dimension)
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # First Conv Block
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Second Conv Block
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Third Conv Block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Fourth Conv Block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history, save_path):
    """
    Plot and save training history

    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def train_model():
    """
    Main training function
    """
    print("Starting training...")

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    label_classes = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'label_classes.npy'))

    num_classes = len(label_classes)

    print(f"\nDataset information:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_classes}")

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

    print(f"\nAfter validation split:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    # Build model
    print("\nBuilding CNN model...")
    model = build_cnn_model(X_train.shape[1], num_classes)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
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

        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f'logs_{timestamp}'),
            histogram_freq=1
        )
    ]

    # Train model
    print("\nStarting training...")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")

    history = model.fit(
        X_train, y_train_cat,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callback_list,
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Plot and save training history
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
        'learning_rate': config.LEARNING_RATE
    }

    config_path = os.path.join(config.RESULTS_DIR, f'training_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)

    print(f"\nTraining configuration saved to: {config_path}")
    print("\nTraining complete!")

    return model, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_STATE)
    tf.random.set_seed(config.RANDOM_STATE)

    # Train model
    model, history = train_model()