"""
Train classifier on emotion2vec Tamil features
This should give us much better results than statistical features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

class EmotionClassifier(nn.Module):
    """Classifier for emotion2vec features"""
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

def train_model(train_loader, val_loader, model, device, num_epochs=100, patience=15):
    """Train the classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    
    print("\nTraining classifier...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
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
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
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
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tamil_emotion2vec_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_tamil_emotion2vec_model.pt'))
    
    return model, best_val_acc

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(5):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100. * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({mask.sum()} samples)")
    
    return accuracy

def main():
    print("=" * 70)
    print("TRAINING TAMIL CLASSIFIER WITH EMOTION2VEC FEATURES")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load emotion2vec features
    feature_dir = '../features/tamil_emotion2vec'
    
    print("Loading features...")
    X_train = np.load(os.path.join(feature_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(feature_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(feature_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(feature_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(feature_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(feature_dir, 'y_test.npy'))
    
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print()
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = EmotionClassifier(input_dim=768, num_classes=5).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Train model
    model, val_acc = train_model(train_loader, val_loader, model, device)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Save model
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'tamil_emotion2vec_classifier.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Compare with baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH ALL APPROACHES")
    print("=" * 70)
    print(f"Simple CNN (baseline):       34.04%")
    print(f"Feature-based (augmented):   36.88%")
    print(f"emotion2vec (frozen):        {test_acc:.2f}%")
    
    improvement = test_acc - 36.88
    print(f"\nImprovement over previous:   {improvement:+.2f}%")
    
    if test_acc > 50:
        print("\n🎉 EXCELLENT! emotion2vec significantly improved results!")
        print("   The pretrained encoder provides much better features")
    elif test_acc > 40:
        print("\n✓ Good improvement! emotion2vec is helping")
    elif test_acc > 36.88:
        print("\n✓ Minor improvement with emotion2vec")
    else:
        print("\n⚠ emotion2vec didn't improve results")
        print("   May need fine-tuning instead of frozen features")

if __name__ == '__main__':
    main()
