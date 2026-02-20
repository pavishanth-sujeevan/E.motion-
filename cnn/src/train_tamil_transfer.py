"""
Train Tamil model using transfer learning from English model.
Freeze early layers, fine-tune later layers.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime
from sklearn.utils import class_weight

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import config
from src import config

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

def create_transfer_model(english_model_path):
    """
    Create a transfer learning model from English model.
    Freeze convolutional layers, train only dense layers.
    """
    print("\n" + "=" * 70)
    print("Loading English Model for Transfer Learning")
    print("=" * 70)
    
    # Load the English model
    print(f"Loading: {english_model_path}")
    base_model = keras.models.load_model(english_model_path)
    print("[OK] English model loaded")
    
    # Display original model
    print("\nOriginal English model:")
    base_model.summary()
    
    # Freeze convolutional layers (keep learned features)
    print("\n" + "=" * 70)
    print("Freezing Convolutional Layers")
    print("=" * 70)
    
    frozen_count = 0
    for layer in base_model.layers:
        # Freeze all Conv2D, BatchNormalization, and MaxPooling layers
        if any(x in layer.name for x in ['conv', 'batch', 'max_pool', 'global']):
            layer.trainable = False
            frozen_count += 1
            print(f"  Frozen: {layer.name}")
        else:
            layer.trainable = True
            print(f"  Trainable: {layer.name}")
    
    print(f"\nTotal frozen layers: {frozen_count}")
    
    # Count trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in base_model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in base_model.weights])
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return base_model

def train_transfer_model():
    """Train the transfer learning model."""
    print("=" * 70)
    print("TRANSFER LEARNING: ENGLISH -> TAMIL")
    print("=" * 70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    # Load English model and prepare for transfer learning
    english_model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'saved_models', 'language_models', 'english', 'english_model.h5'
    )
    
    model = create_transfer_model(english_model_path)
    
    # Recompile with lower learning rate for fine-tuning
    fine_tune_lr = config.LEARNING_RATE * 0.1  # 10x smaller
    print(f"\nUsing fine-tuning learning rate: {fine_tune_lr}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
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
    
    checkpoint_path = os.path.join(model_dir, f'tamil_transfer_{timestamp}.h5')
    
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
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 70)
    print("Transfer Learning Configuration")
    print("=" * 70)
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {fine_tune_lr}")
    print(f"Early stopping patience: 20 epochs")
    print(f"Using class weights: Yes")
    print(f"Strategy: Freeze conv layers, fine-tune dense layers")
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
    final_model_path = os.path.join(model_dir, 'tamil_transfer_model.h5')
    model.save(final_model_path)
    print(f"\nFinal Tamil transfer model saved to: {final_model_path}")
    
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model: {checkpoint_path}")
    print(f"Final model: {final_model_path}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("\nComparison:")
    print(f"  Original Tamil model: 12.06%")
    print(f"  Transfer learning:    {test_accuracy*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    train_transfer_model()
