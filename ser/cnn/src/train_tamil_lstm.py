"""
Train Tamil Emotion Recognition Model with LSTM + Attention
Uses Bidirectional LSTM with attention mechanism for temporal pattern recognition
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import sys
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

TARGET_EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']

# Attention Layer
class AttentionLayer(layers.Layer):
    """
    Attention mechanism to focus on important time frames in audio
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[-1],),
                                initializer='zeros',
                                trainable=True)
        self.u = self.add_weight(name='attention_vector',
                                shape=(input_shape[-1],),
                                initializer='glorot_uniform',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch, time_steps, features)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        weighted_input = x * tf.expand_dims(a, -1)
        output = tf.reduce_sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_lstm_attention_model(input_shape, num_classes):
    """
    Build LSTM model with attention mechanism
    
    Architecture:
    - 2 Bidirectional LSTM layers with dropout
    - Attention mechanism to focus on important frames
    - Dense layers with regularization
    - Total params: ~150K (suitable for limited data)
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # First Bi-LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bi_lstm_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    
    # Second Bi-LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bi_lstm_2'
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    x = AttentionLayer(name='attention')(x)
    x = layers.Dropout(0.4)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='lstm_attention_model')
    return model

def load_tamil_data():
    """Load preprocessed Tamil data"""
    print("Loading Tamil data...")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_tamil')
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"X shape: {X_train.shape}")
    
    # Reshape X for LSTM: (samples, time_steps, features)
    # Current: (samples, n_mels, time_steps, 1)
    # Need: (samples, time_steps, n_mels)
    # Remove channel dimension and transpose
    X_train = np.squeeze(X_train, axis=-1)  # (samples, n_mels, time_steps)
    X_val = np.squeeze(X_val, axis=-1)
    X_test = np.squeeze(X_test, axis=-1)
    
    X_train = np.transpose(X_train, (0, 2, 1))  # (samples, time_steps, n_mels)
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    print(f"Reshaped X for LSTM: {X_train.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main training function"""
    print("=" * 70)
    print("TAMIL EMOTION RECOGNITION - LSTM + ATTENTION MODEL")
    print("=" * 70)
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_tamil_data()
    
    # Print emotion distribution
    print("\nEmotion distribution (training set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for emotion_idx, count in zip(unique, counts):
        emotion_name = TARGET_EMOTIONS[emotion_idx]
        print(f"  {emotion_name}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\nClass weights:", class_weight_dict)
    
    # Build model
    print("\nBuilding LSTM + Attention model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, n_mels)
    model = build_lstm_attention_model(input_shape, len(TARGET_EMOTIONS))
    
    # Compile model with lower learning rate for LSTM
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Calculate total parameters
    trainable_params = sum([np.prod(var.shape) for var in model.trainable_variables])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    
    # Callbacks
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'models', 'saved_models', 'language_models', 'tamil')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'tamil_lstm_model.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    print("-" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Per-class evaluation
    print("\nPer-Class Performance:")
    print("-" * 50)
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    for i, emotion in enumerate(TARGET_EMOTIONS):
        mask = y_test == i
        if mask.sum() > 0:
            correct = (y_pred[mask] == i).sum()
            total = mask.sum()
            accuracy = correct / total * 100
            print(f"{emotion.capitalize():10s}: {correct:3d}/{total:3d} = {accuracy:5.1f}%")
    
    # Save training history
    log_dir = os.path.dirname(os.path.dirname(__file__))
    log_file = os.path.join(log_dir, 'training_lstm_tamil.log')
    with open(log_file, 'w') as f:
        f.write("TAMIL LSTM + ATTENTION MODEL TRAINING LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model: LSTM + Attention\n")
        f.write(f"Parameters: {trainable_params:,}\n")
        f.write(f"Architecture: 2x Bi-LSTM (64, 32) + Attention + Dense (64, 32)\n\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(f"Final Test Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n\n")
        f.write("Per-Class Performance:\n")
        for i, emotion in enumerate(TARGET_EMOTIONS):
            mask = y_test == i
            if mask.sum() > 0:
                correct = (y_pred[mask] == i).sum()
                total = mask.sum()
                accuracy = correct / total * 100
                f.write(f"  {emotion.capitalize():10s}: {accuracy:5.1f}%\n")
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Training log saved to: {log_file}")
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
