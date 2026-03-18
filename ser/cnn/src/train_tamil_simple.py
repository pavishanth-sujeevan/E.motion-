"""
Train a simpler CNN model on Tamil data (fewer parameters).
This lighter model should work better with limited data.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime
from sklearn.utils import class_weight

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import config
from src import config

def build_simple_cnn(input_shape, num_classes):
    """
    Build a simpler CNN with fewer parameters.
    Original model: ~1.27M parameters
    This model: ~200K parameters
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1 - 32 filters
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Conv Block 2 - 64 filters
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Conv Block 3 - 128 filters (only 1 more block)
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers - smaller
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_preprocessed_data():
    """Load preprocessed Tamil data."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_tamil')
    
    print("Loading preprocessed Tamil data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_simple_model():
    """Train the simple model."""
    print("=" * 70)
    print("TRAINING SIMPLE CNN ON TAMIL DATA")
    print("=" * 70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    # Build model
    input_shape = X_train.shape[1:]
    num_classes = len(config.EMOTIONS)
    
    print("\n" + "=" * 70)
    print("Building Simple CNN Model")
    print("=" * 70)
    
    model = build_simple_cnn(input_shape, num_classes)
    
    # Display model summary
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate class weights
    class_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
    
    print("\nClass weights:")
    for emotion, idx in config.EMOTION_TO_INDEX.items():
        if idx in class_weights_dict:
            print(f"  {emotion:10s}: {class_weights_dict[idx]:.3f}")
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'models', 'saved_models', 'language_models', 'tamil')
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, f'tamil_simple_{timestamp}.h5')
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # Reduced from 25
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,  # Reduced from 10
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Early stopping patience: 20 epochs")
    print(f"Using class weights: Yes")
    print("=" * 70)
    print()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nPer-class test accuracy:")
    for emotion, idx in config.EMOTION_TO_INDEX.items():
        mask = y_test == idx
        if mask.sum() > 0:
            correct = (y_pred_classes[mask] == idx).sum()
            total = mask.sum()
            accuracy = correct / total
            print(f"  {emotion:12s}: {accuracy:.4f} ({accuracy*100:5.1f}%) - {correct}/{total} correct")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'tamil_simple_model.h5')
    model.save(final_model_path)
    print(f"\nFinal Tamil simple model saved to: {final_model_path}")
    
    print("\n" + "=" * 70)
    print("SIMPLE TAMIL MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model: {checkpoint_path}")
    print(f"Final model: {final_model_path}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 70)

if __name__ == "__main__":
    train_simple_model()
