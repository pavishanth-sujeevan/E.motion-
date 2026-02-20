"""
Training script for Tamil-only CNN model
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
from sklearn.utils.class_weight import compute_class_weight

import config


def build_spectrogram_cnn(input_shape, num_classes):
    """Build CNN model for mel spectrogram classification"""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.close()


def train_tamil_model():
    """Train Tamil-only model"""
    print("="*70)
    print("TRAINING TAMIL-ONLY MEL SPECTROGRAM CNN")
    print("="*70)

    # Load processed data
    print("\nLoading preprocessed Tamil spectrograms...")
    tamil_data_dir = os.path.join(config.DATA_DIR, 'processed_tamil')

    X_train = np.load(os.path.join(tamil_data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(tamil_data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(tamil_data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(tamil_data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(tamil_data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(tamil_data_dir, 'y_test.npy'))

    num_classes = len(config.EMOTIONS)

    # Dataset info
    print(f"\n{'='*70}")
    print("DATASET INFORMATION")
    print(f"{'='*70}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {config.EMOTIONS}")

    # Class distribution
    print(f"\nTraining set class distribution:")
    for idx in range(num_classes):
        count = (y_train == idx).sum()
        emotion = config.INDEX_TO_EMOTION[idx]
        percentage = (count / len(y_train)) * 100
        print(f"  {emotion:12s}: {count:4d} ({percentage:5.1f}%)")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\nClass weights (to handle imbalance):")
    for idx, weight in class_weight_dict.items():
        emotion = config.INDEX_TO_EMOTION[idx]
        print(f"  {emotion:12s}: {weight:.4f}")

    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Build model
    print(f"\n{'='*70}")
    print("BUILDING CNN MODEL")
    print(f"{'='*70}")

    model = build_spectrogram_cnn(X_train.shape[1:], num_classes)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create Tamil models directory
    tamil_models_dir = os.path.join(config.MODELS_DIR, 'language_models', 'tamil')
    os.makedirs(tamil_models_dir, exist_ok=True)
    
    model_path = os.path.join(tamil_models_dir, f'tamil_model_{timestamp}.h5')

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),

        callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f'tamil_logs_{timestamp}'),
            histogram_freq=1
        )
    ]

    # Train
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Early stopping patience: 25 epochs")
    print(f"Using class weights: Yes")
    print(f"{'='*70}\n")

    history = model.fit(
        X_train, y_train_cat,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callback_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*70}")

    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\nPer-class test accuracy:")
    for idx in range(num_classes):
        emotion = config.INDEX_TO_EMOTION[idx]
        mask = y_test == idx
        if mask.sum() > 0:
            class_correct = (y_pred_classes[mask] == idx).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total
            print(f"  {emotion:12s}: {class_acc:.4f} ({class_acc*100:5.1f}%) - {class_correct}/{class_total} correct")

    # Save final model
    final_model_path = os.path.join(tamil_models_dir, 'tamil_model.h5')
    model.save(final_model_path)
    print(f"\nFinal Tamil model saved to: {final_model_path}")

    # Plot history
    tamil_results_dir = os.path.join(config.RESULTS_DIR, 'tamil_training')
    os.makedirs(tamil_results_dir, exist_ok=True)
    history_plot_path = os.path.join(tamil_results_dir, f'training_history_{timestamp}.png')
    plot_training_history(history, history_plot_path)

    # Save configuration
    training_config = {
        'timestamp': timestamp,
        'model_type': 'tamil_mel_spectrogram_cnn',
        'language': 'Tamil',
        'dataset': 'EMOTA',
        'input_shape': list(X_train.shape[1:]),
        'num_classes': num_classes,
        'classes': config.EMOTIONS,
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'epochs_trained': len(history.history['loss']),
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'total_parameters': int(total_params)
    }

    config_path = os.path.join(tamil_results_dir, f'training_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)

    print(f"Training configuration saved to: {config_path}")

    print(f"\n{'='*70}")
    print("TAMIL MODEL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nBest model: {model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\nNext step: Test the Tamil model")
    print(f"  python test_tamil_model.py")
    print(f"{'='*70}\n")

    return model, history


if __name__ == "__main__":
    # Set seeds
    np.random.seed(config.RANDOM_STATE)
    tf.random.set_seed(config.RANDOM_STATE)

    # Train
    model, history = train_tamil_model()
