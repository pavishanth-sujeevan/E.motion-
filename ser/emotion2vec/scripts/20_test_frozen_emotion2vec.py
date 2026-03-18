"""
Test emotion2vec base model (frozen) on English and Tamil data
No fine-tuning - just use pretrained features with a simple classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

# ===== Simple Classifier =====

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, language):
    """Train a simple classifier on frozen features"""
    
    print(f"\n{'='*70}")
    print(f"TESTING FROZEN EMOTION2VEC ON {language.upper()}")
    print('='*70)
    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create classifier
    classifier = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    print("\nTraining classifier on frozen features...")
    
    for epoch in range(100):
        classifier.train()
        
        # Train
        optimizer.zero_grad()
        outputs = classifier(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(X_val)
            _, val_pred = val_outputs.max(1)
            val_acc = 100. * (val_pred == y_val).sum().item() / len(y_val)
            
            train_outputs = classifier(X_train)
            _, train_pred = train_outputs.max(1)
            train_acc = 100. * (train_pred == y_train).sum().item() / len(y_train)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Train={train_acc:.2f}% Val={val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = classifier.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and test
    classifier.load_state_dict(best_state)
    classifier.eval()
    
    with torch.no_grad():
        test_outputs = classifier(X_test)
        _, test_pred = test_outputs.max(1)
        test_acc = 100. * (test_pred == y_test).sum().item() / len(y_test)
    
    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS - {language.upper()}")
    print('='*70)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc:.2f}%\n")
    
    # Per-class accuracy
    print("Per-class test accuracy:")
    for i in range(5):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = 100. * (test_pred[mask] == y_test[mask]).sum().item() / mask.sum().item()
            count = mask.sum().item()
            print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({count:3d} samples)")
    
    # Confusion info
    print(f"\nPrediction distribution:")
    for i in range(5):
        pred_count = (test_pred == i).sum().item()
        print(f"  Predicted {INDEX_TO_EMOTION[i]:8s}: {pred_count:3d} times")
    
    return test_acc

def main():
    print("=" * 70)
    print("TESTING FROZEN EMOTION2VEC BASE MODEL")
    print("=" * 70)
    print("\nThis tests the pretrained emotion2vec encoder WITHOUT fine-tuning.")
    print("We only train a simple classifier head on top of frozen features.\n")
    
    results = {}
    
    # ===== TEST ON ENGLISH =====
    print("\n" + "█" * 70)
    print("TEST 1: ENGLISH DATA (5,664 samples)")
    print("█" * 70)
    
    feature_dir = Path('../features/english_emotion2vec')
    if feature_dir.exists():
        X_train = np.load(feature_dir / 'X_train.npy')
        y_train = np.load(feature_dir / 'y_train.npy')
        X_val = np.load(feature_dir / 'X_val.npy')
        y_val = np.load(feature_dir / 'y_val.npy')
        X_test = np.load(feature_dir / 'X_test.npy')
        y_test = np.load(feature_dir / 'y_test.npy')
        
        results['English'] = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test, 'English'
        )
    else:
        print("\n❌ English features not found. Run script 16 first.")
        results['English'] = None
    
    # ===== TEST ON TAMIL =====
    print("\n" + "█" * 70)
    print("TEST 2: TAMIL DATA (936 samples)")
    print("█" * 70)
    
    feature_dir = Path('../features/tamil_emotion2vec')
    if feature_dir.exists():
        X_train = np.load(feature_dir / 'X_train.npy')
        y_train = np.load(feature_dir / 'y_train.npy')
        X_val = np.load(feature_dir / 'X_val.npy')
        y_val = np.load(feature_dir / 'y_val.npy')
        X_test = np.load(feature_dir / 'X_test.npy')
        y_test = np.load(feature_dir / 'y_test.npy')
        
        results['Tamil'] = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test, 'Tamil'
        )
    else:
        print("\n❌ Tamil features not found. Run script 12 first.")
        results['Tamil'] = None
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - FROZEN EMOTION2VEC BASE MODEL")
    print("=" * 70)
    print()
    
    if results['English'] is not None:
        print(f"English (5,664 samples):")
        print(f"  emotion2vec frozen:  {results['English']:6.2f}%")
        print(f"  CNN baseline:        94.89%")
        print(f"  Gap:                 {94.89 - results['English']:.2f}%")
        print()
    
    if results['Tamil'] is not None:
        print(f"Tamil (936 samples):")
        print(f"  emotion2vec frozen:  {results['Tamil']:6.2f}%")
        print(f"  Simple CNN:          34.04%")
        print(f"  Feature-based MLP:   36.88%")
        print()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if results['English'] is not None and results['English'] < 50:
        print("\n❌ emotion2vec frozen features perform POORLY on emotion recognition.")
        print("   The pretrained model was trained on general speech representation,")
        print("   not emotion-specific features.")
        print()
        print("   Recommendation: Use CNN for English, Feature-based MLP for Tamil.")
    elif results['English'] is not None and results['English'] > 80:
        print("\n✅ emotion2vec frozen features work well!")
        print("   The pretrained features capture emotion information.")
    else:
        print("\n⚠️  emotion2vec shows some promise but needs improvement.")
        print("    Fine-tuning or more data may help.")
    
    print()

if __name__ == '__main__':
    main()
