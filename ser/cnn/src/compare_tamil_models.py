"""
Compare all three Tamil models: Original, Simple, and Transfer Learning
"""
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import config
from src import config

def load_test_data():
    """Load preprocessed Tamil test data."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_tamil')
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_test, y_test

def evaluate_model(model_path, model_name):
    """Evaluate a model on test data."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found!")
        return None
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Recompile to ensure consistent loss function
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[OK] Model loaded")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    per_class_results = {}
    for emotion, idx in config.EMOTION_TO_INDEX.items():
        mask = y_test == idx
        if mask.sum() > 0:
            correct = (y_pred_classes[mask] == idx).sum()
            total = mask.sum()
            accuracy = correct / total
            per_class_results[emotion] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
    
    return {
        'model_name': model_name,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'per_class': per_class_results
    }

def compare_models():
    """Compare all three models."""
    print("="*70)
    print("TAMIL MODEL COMPARISON")
    print("="*70)
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'models', 'saved_models', 'language_models', 'tamil')
    
    models = [
        {
            'path': os.path.join(models_dir, 'tamil_model.h5'),
            'name': 'Original Deep CNN (from scratch)',
            'params': '~1.27M parameters'
        },
        {
            'path': os.path.join(models_dir, 'tamil_simple_model.h5'),
            'name': 'Simple CNN (fewer layers)',
            'params': '~118K parameters'
        },
        {
            'path': os.path.join(models_dir, 'tamil_transfer_model.h5'),
            'name': 'Transfer Learning (from English)',
            'params': '~1.27M parameters (frozen conv layers)'
        }
    ]
    
    results = []
    for model_info in models:
        result = evaluate_model(model_info['path'], model_info['name'])
        if result:
            result['params'] = model_info['params']
            results.append(result)
    
    # Print comparison table
    print("\n" + "="*70)
    print("OVERALL RESULTS COMPARISON")
    print("="*70)
    print(f"{'Model':<40} {'Accuracy':>12} {'Loss':>12}")
    print("-"*70)
    
    for result in results:
        print(f"{result['model_name']:<40} {result['test_accuracy']*100:>11.2f}% {result['test_loss']:>12.4f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print("-"*70)
    print(f"{'BEST MODEL: ' + best_result['model_name']:<40} {best_result['test_accuracy']*100:>11.2f}%")
    print("="*70)
    
    # Per-emotion comparison
    print("\n" + "="*70)
    print("PER-EMOTION ACCURACY COMPARISON")
    print("="*70)
    
    emotions = config.EMOTIONS
    print(f"{'Emotion':<12}", end='')
    for result in results:
        print(f"{result['model_name'][:20]:>22}", end='')
    print()
    print("-"*70)
    
    for emotion in emotions:
        print(f"{emotion:<12}", end='')
        for result in results:
            if emotion in result['per_class']:
                acc = result['per_class'][emotion]['accuracy']
                print(f"{acc*100:>20.1f}%", end='  ')
            else:
                print(f"{'N/A':>20}", end='  ')
        print()
    
    print("="*70)
    
    # Detailed per-class for best model
    print("\n" + "="*70)
    print(f"DETAILED RESULTS - {best_result['model_name']}")
    print("="*70)
    print(f"Overall Test Accuracy: {best_result['test_accuracy']*100:.2f}%")
    print(f"Overall Test Loss: {best_result['test_loss']:.4f}")
    print(f"\nPer-emotion breakdown:")
    
    for emotion in emotions:
        if emotion in best_result['per_class']:
            data = best_result['per_class'][emotion]
            acc = data['accuracy']
            correct = data['correct']
            total = data['total']
            print(f"  {emotion:<10}: {acc*100:5.1f}% ({correct:2d}/{total:2d} correct)")
    
    print("="*70)
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    accuracies = [r['test_accuracy']*100 for r in results]
    improvement = max(accuracies) - min(accuracies)
    
    print(f"\n1. Performance Range: {min(accuracies):.1f}% - {max(accuracies):.1f}%")
    print(f"   Improvement: +{improvement:.1f}% from worst to best")
    
    print(f"\n2. Winner: {best_result['model_name']}")
    print(f"   Accuracy: {best_result['test_accuracy']*100:.2f}%")
    print(f"   Architecture: {best_result['params']}")
    
    print("\n3. Key Findings:")
    if best_result['model_name'] == 'Simple CNN (fewer layers)':
        print("   - Simpler architecture works better with limited data (936 samples)")
        print("   - Fewer parameters reduce overfitting risk")
        print("   - 118K parameters vs 1.27M in original model")
    elif best_result['model_name'] == 'Transfer Learning (from English)':
        print("   - English features transfer well to Tamil")
        print("   - Pre-trained weights provide better starting point")
        print("   - Frozen conv layers prevent overfitting")
    
    print("\n4. Next Steps:")
    if best_result['test_accuracy'] < 0.5:
        print("   - Current accuracy still low (<50%)")
        print("   - RECOMMENDED: Apply data augmentation to increase dataset size")
        print("   - Consider collecting more Tamil samples")
        print("   - Try ensemble methods combining multiple models")
    elif best_result['test_accuracy'] < 0.7:
        print("   - Moderate performance (50-70%)")
        print("   - Data augmentation could help reach 70%+ target")
        print("   - Fine-tune hyperparameters (learning rate, dropout)")
    else:
        print("   - Good performance! (>70%)")
        print("   - Model ready for deployment")
        print("   - Consider additional validation with real-world data")
    
    print("="*70)

if __name__ == "__main__":
    compare_models()
