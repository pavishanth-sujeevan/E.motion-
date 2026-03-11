"""
Train Simple CNN on augmented Tamil data
This is the most practical approach given our constraints
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import sys

# Emotion mappings (avoid slow config import)
EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

def create_simple_cnn_model(input_shape, num_classes=5):
    """
    Create Simple CNN model (best performing for Tamil with limited data)
    118K parameters - prevents overfitting
    """
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_augmented_data():
    """Load augmented Tamil dataset"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cnn', 'data', 'processed_tamil_augmented')
    
    print("Loading augmented data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print()
    
    # Convert labels to one-hot
    y_train_cat = to_categorical(y_train, num_classes=5)
    y_val_cat = to_categorical(y_val, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)
    
    return (X_train, y_train_cat, y_train), (X_val, y_val_cat, y_val), (X_test, y_test_cat, y_test)

def evaluate_model(model, X_test, y_test_cat, y_test_labels):
    """Evaluate model and show per-class metrics"""
    # Overall accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)
    print(f"\nOverall Test Accuracy: {test_acc * 100:.2f}%")
    print()
    
    # Per-class accuracy
    print("Per-class accuracy:")
    for emotion_idx in range(5):
        emotion_name = INDEX_TO_EMOTION[emotion_idx]
        mask = y_test_labels == emotion_idx
        
        if mask.sum() > 0:
            class_correct = (y_pred[mask] == y_test_labels[mask]).sum()
            class_total = mask.sum()
            class_acc = 100. * class_correct / class_total
            print(f"  {emotion_name:8s}: {class_acc:6.2f}% ({class_total} samples)")
    
    return test_acc * 100

def main():
    print("=" * 70)
    print("TRAINING SIMPLE CNN ON AUGMENTED TAMIL DATA")
    print("=" * 70)
    print()
    
    # Load data
    (X_train, y_train_cat, y_train_labels), (X_val, y_val_cat, y_val_labels), (X_test, y_test_cat, y_test_labels) = load_augmented_data()
    
    # Create model
    print("Creating Simple CNN model...")
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    model = create_simple_cnn_model(input_shape)
    
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Callbacks
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'tamil_augmented_cnn.h5')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("Training model...")
    print("-" * 70)
    
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate
    test_acc = evaluate_model(model, X_test, y_test_cat, y_test_labels)
    
    # Compare with baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("=" * 70)
    print(f"Simple CNN (original):        34.04%")
    print(f"Feature-based (augmented):    36.88%")
    print(f"Simple CNN (augmented):       {test_acc:.2f}%")
    
    improvement = test_acc - 34.04
    print(f"\nImprovement over baseline:    {improvement:+.2f}%")
    
    if test_acc > 40:
        print("\n✓ Excellent! Augmentation significantly improved performance!")
    elif test_acc > 36.88:
        print("\n✓ Good! CNN outperformed feature-based approach")
    else:
        print("\n⚠ Model did not improve as expected")
    
    print(f"\n✓ Model saved to: {model_path}")

if __name__ == '__main__':
    main()
