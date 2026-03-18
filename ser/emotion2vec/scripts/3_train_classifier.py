"""
Train emotion classifier using extracted features
Approach 1: Simple classifier on extracted features (fast baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import TARGET_EMOTIONS

class EmotionClassifier(nn.Module):
    """Simple MLP classifier for emotion recognition"""
    def __init__(self, input_dim=100, num_classes=5):
        super(EmotionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def load_features(feature_dir):
    """Load extracted features"""
    X_train = np.load(os.path.join(feature_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(feature_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(feature_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(feature_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(feature_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(feature_dir, 'y_test.npy'))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=15):
    """Train the model with early stopping"""
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_acc

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=TARGET_EMOTIONS))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, emotion in enumerate(TARGET_EMOTIONS):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100 * (all_preds[mask] == i).sum() / mask.sum()
            print(f"  {emotion.capitalize():10s}: {class_acc:.2f}%")
    
    return accuracy, all_preds, all_labels

def train_language_model(language, feature_dir, model_save_dir):
    """Train model for specific language"""
    print("=" * 70)
    print(f"TRAINING {language.upper()} MODEL")
    print("=" * 70)
    print()
    
    # Load features
    X_train, X_val, X_test, y_train, y_val, y_test = load_features(feature_dir)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print()
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
    class_weights = torch.FloatTensor(class_weights)
    print("Class weights:", class_weights.tolist())
    print()
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = EmotionClassifier(input_dim=X_train.shape[1], num_classes=len(TARGET_EMOTIONS))
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Train
    print("Starting training...")
    print("-" * 70)
    best_val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=15)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    test_acc, test_preds, test_labels = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Save model
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f'{language}_classifier.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc
    }, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return test_acc

def main():
    print()
    print("=" * 70)
    print("EMOTION CLASSIFIER TRAINING")
    print("=" * 70)
    print()
    
    # Paths
    features_dir = os.path.join(os.path.dirname(__file__), '..', 'features')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    results = {}
    
    # Train English model
    english_features = os.path.join(features_dir, 'english')
    if os.path.exists(english_features):
        english_acc = train_language_model('english', english_features, models_dir)
        results['english'] = english_acc
    else:
        print(f"⚠ English features not found at: {english_features}")
    
    # Train Tamil model
    print("\n\n")
    tamil_features = os.path.join(features_dir, 'tamil')
    if os.path.exists(tamil_features):
        tamil_acc = train_language_model('tamil', tamil_features, models_dir)
        results['tamil'] = tamil_acc
    else:
        print(f"⚠ Tamil features not found at: {tamil_features}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for lang, acc in results.items():
        print(f"{lang.capitalize():10s}: {acc:.2f}%")
    
    print("\nNext steps:")
    print("  1. Compare these results with CNN models")
    print("  2. Try Approach 2 (partial fine-tuning) if results are promising")

if __name__ == '__main__':
    main()
