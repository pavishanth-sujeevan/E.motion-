"""
Compare all 4 Tamil emotion recognition models:
1. Deep CNN (original)
2. Simple CNN 
3. Transfer Learning
4. LSTM + Attention
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import config

# Custom attention layer for loading LSTM model
class AttentionLayer(keras.layers.Layer):
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
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        weighted_input = x * tf.expand_dims(a, -1)
        output = tf.reduce_sum(weighted_input, axis=1)
        return output

def load_test_data():
    """Load test data for evaluation"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_tamil')
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_test, y_test

def load_test_data_lstm():
    """Load and reshape test data for LSTM"""
    X_test, y_test = load_test_data()
    # Reshape for LSTM: remove channel and transpose
    X_test = np.squeeze(X_test, axis=-1)
    X_test = np.transpose(X_test, (0, 2, 1))
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return detailed results"""
    print(f"\nEvaluating {model_name}...")
    print("-" * 70)
    
    # Overall accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Per-class performance
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    print("\nPer-Class Performance:")
    class_accuracies = []
    for i, emotion in enumerate(config.EMOTIONS):
        mask = y_test == i
        if mask.sum() > 0:
            correct = (y_pred[mask] == i).sum()
            total = mask.sum()
            accuracy = correct / total * 100
            class_accuracies.append(accuracy)
            print(f"  {emotion.capitalize():10s}: {correct:3d}/{total:3d} = {accuracy:5.1f}%")
        else:
            class_accuracies.append(0.0)
            print(f"  {emotion.capitalize():10s}: No samples")
    
    return test_accuracy * 100, test_loss, class_accuracies

def main():
    print("=" * 70)
    print("TAMIL MODEL COMPARISON - ALL 4 APPROACHES")
    print("=" * 70)
    print()
    
    # Model paths
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'models', 'saved_models', 'language_models', 'tamil')
    
    models = [
        ('Deep CNN (Original)', 'tamil_model.h5', False),
        ('Simple CNN', 'tamil_simple_model.h5', False),
        ('Transfer Learning', 'tamil_transfer_model.h5', False),
        ('LSTM + Attention', 'tamil_lstm_model.h5', True),
    ]
    
    # Load test data (CNN format)
    X_test_cnn, y_test = load_test_data()
    print(f"Test samples: {len(y_test)}")
    print(f"CNN data shape: {X_test_cnn.shape}")
    
    # Load test data (LSTM format)
    X_test_lstm, _ = load_test_data_lstm()
    print(f"LSTM data shape: {X_test_lstm.shape}\n")
    
    results = []
    
    for model_name, model_file, is_lstm in models:
        try:
            model_path = os.path.join(models_dir, model_file)
            
            if not os.path.exists(model_path):
                print(f"\n{model_name}: Model file not found - {model_file}")
                results.append((model_name, 0.0, 0.0, [0.0]*5))
                continue
            
            # Load model with custom objects if needed
            if is_lstm:
                model = keras.models.load_model(
                    model_path,
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
                X_test = X_test_lstm
            else:
                model = keras.models.load_model(model_path)
                X_test = X_test_cnn
            
            # Recompile to ensure consistent evaluation
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Evaluate
            accuracy, loss, class_accs = evaluate_model(model, X_test, y_test, model_name)
            results.append((model_name, accuracy, loss, class_accs))
            
        except Exception as e:
            print(f"\n{model_name}: Error loading model - {str(e)}")
            results.append((model_name, 0.0, 0.0, [0.0]*5))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print()
    
    print("Overall Test Accuracy:")
    print("-" * 50)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (name, acc, loss, _) in enumerate(sorted_results, 1):
        marker = " <- BEST" if rank == 1 else ""
        print(f"{rank}. {name:25s}: {acc:5.2f}%{marker}")
    
    print("\n\nPer-Emotion Comparison:")
    print("-" * 70)
    print(f"{'Emotion':<12} {'Deep CNN':<12} {'Simple CNN':<12} {'Transfer':<12} {'LSTM':<12}")
    print("-" * 70)
    
    for i, emotion in enumerate(config.EMOTIONS):
        row = f"{emotion.capitalize():<12}"
        for _, _, _, class_accs in results:
            row += f" {class_accs[i]:5.1f}%      "
        print(row)
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best_model, best_acc, _, _ = sorted_results[0]
    print(f"\nBest Model: {best_model}")
    print(f"Accuracy: {best_acc:.2f}%")
    print()
    print("Model Selection Guide:")
    print("  - Simple CNN: Best for limited data (<1000 samples)")
    print("  - Deep CNN: Best for large datasets (>2000 samples)")
    print("  - Transfer Learning: Best when source/target languages are similar")
    print("  - LSTM: Requires more data (>3000 samples) for temporal learning")
    print()
    print("For Tamil (936 samples): Use Simple CNN model")
    print("Model file: tamil_simple_model.h5")

if __name__ == '__main__':
    main()
