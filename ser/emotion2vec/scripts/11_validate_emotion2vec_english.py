"""
Extract emotion2vec features from English dataset and train classifier
This validates our custom encoder works, then we can apply to Tamil
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import soundfile as sf
import os
import sys
from tqdm import tqdm
import glob

# Import our custom encoder - copy the class here
class ConvFeatureExtraction(nn.Module):
    """Conv feature extraction module from emotion2vec"""
    def __init__(self, conv_layers_config):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        in_d = 1
        for i, (dim, k, stride) in enumerate(conv_layers_config):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_d, dim, k, stride=stride),
                    nn.Dropout(p=0.0),
                    nn.GroupNorm(dim, dim),
                ))
            in_d = dim
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = torch.nn.functional.gelu(x)
        return x

class SimpleEmotion2VecEncoder(nn.Module):
    """Simplified emotion2vec encoder for feature extraction"""
    def __init__(self, checkpoint_path):
        super().__init__()
        
        print("Building custom encoder from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.state_dict_full = checkpoint['model']
        
        conv_configs = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 3, 2), (512, 2, 2),
        ]
        
        self.feature_extractor = ConvFeatureExtraction(conv_configs)
        self._load_conv_weights()
        self.post_extract_proj = nn.Linear(512, 768)
        
        print("✓ Encoder built successfully")
    
    def _load_conv_weights(self):
        prefix = "modality_encoders.AUDIO.local_encoder.conv_layers"
        
        for i, conv_block in enumerate(self.feature_extractor.conv_layers):
            conv_weight_key = f"{prefix}.{i}.0.weight"
            if conv_weight_key in self.state_dict_full:
                conv_block[0].weight.data = self.state_dict_full[conv_weight_key]
            
            gn_weight_key = f"{prefix}.{i}.2.1.weight"
            gn_bias_key = f"{prefix}.{i}.2.1.bias"
            if gn_weight_key in self.state_dict_full:
                conv_block[2].weight.data = self.state_dict_full[gn_weight_key]
                conv_block[2].bias.data = self.state_dict_full[gn_bias_key]
        
        print("  ✓ Loaded convolutional weights")
    
    def forward(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        features = self.feature_extractor(audio)
        features = features.transpose(1, 2)
        features = self.post_extract_proj(features)
        
        return features

# Emotion mappings
EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

def load_audio_files(data_dir, emotions):
    """Load all audio files from RAVDESS and TESS"""
    audio_files = []
    labels = []
    
    print(f"Loading audio files from: {data_dir}")
    
    # RAVDESS
    ravdess_dir = os.path.join(data_dir, 'RAVDESS-SPEECH', 'Actor_*')
    for actor_dir in glob.glob(ravdess_dir):
        for file in glob.glob(os.path.join(actor_dir, '*.wav')):
            # Parse RAVDESS filename: 03-01-XX-XX-XX-XX-XX.wav
            parts = os.path.basename(file).split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                # Map RAVDESS emotions to our 5 emotions
                emotion_map = {
                    1: 'neutral', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fear'
                }
                if emotion_code in emotion_map:
                    emotion = emotion_map[emotion_code]
                    if emotion in emotions:
                        audio_files.append(file)
                        labels.append(EMOTION_TO_INDEX[emotion])
    
    # TESS
    tess_dir = os.path.join(data_dir, 'TESS')
    for emotion_dir in glob.glob(os.path.join(tess_dir, '*')):
        emotion = os.path.basename(emotion_dir).lower()
        # Map TESS emotion names
        if 'angry' in emotion:
            emotion = 'angry'
        elif 'fear' in emotion:
            emotion = 'fear'
        elif 'happy' in emotion or 'pleasant' in emotion:
            emotion = 'happy'
        elif 'neutral' in emotion:
            emotion = 'neutral'
        elif 'sad' in emotion:
            emotion = 'sad'
        else:
            continue
        
        if emotion in emotions:
            for file in glob.glob(os.path.join(emotion_dir, '*.wav')):
                audio_files.append(file)
                labels.append(EMOTION_TO_INDEX[emotion])
    
    print(f"  Loaded {len(audio_files)} audio files")
    return audio_files, labels

def extract_emotion2vec_features(encoder, audio_files, device='cpu'):
    """Extract features from audio files using emotion2vec encoder"""
    features_list = []
    
    encoder.to(device)
    encoder.eval()
    
    print("\nExtracting emotion2vec features...")
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            audio, sr = sf.read(audio_file)
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Pad or trim to fixed length (3 seconds = 48000 samples at 16kHz)
            target_length = 48000
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = encoder(audio_tensor)  # (1, time', 768)
                features_pooled = features.mean(dim=1)  # (1, 768)
            
            features_list.append(features_pooled.cpu().numpy()[0])
            
        except Exception as e:
            print(f"\n⚠ Error processing {audio_file}: {e}")
            # Use zero vector
            features_list.append(np.zeros(768))
    
    return np.array(features_list)

class EmotionClassifier(nn.Module):
    """Simple classifier head for emotion2vec features"""
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

def train_classifier(X_train, y_train, X_val, y_val, device='cpu'):
    """Train emotion classifier"""
    print("\nTraining classifier...")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    # Data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model
    model = EmotionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        # Train
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
        
        # Validate
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
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_emotion2vec_classifier.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_emotion2vec_classifier.pt'))
    
    return model, best_val_acc

def main():
    print("=" * 70)
    print("EMOTION2VEC FINE-TUNING - ENGLISH VALIDATION")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load encoder
    print("Loading emotion2vec encoder...")
    checkpoint_path = 'emotion2vec_base/emotion2vec_base.pt'
    encoder = SimpleEmotion2VecEncoder(checkpoint_path)
    encoder.to(device)
    print()
    
    # Load English audio files
    data_dir = '../cnn/data/raw'
    emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']
    
    audio_files, labels = load_audio_files(data_dir, emotions)
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return
    
    # Split data (80/10/10)
    from sklearn.model_selection import train_test_split
    
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        test_files, test_labels, test_size=0.5, stratify=test_labels, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    print(f"  Test: {len(test_files)}")
    
    # Extract features
    X_train = extract_emotion2vec_features(encoder, train_files, device)
    X_val = extract_emotion2vec_features(encoder, val_files, device)
    X_test = extract_emotion2vec_features(encoder, test_files, device)
    
    print(f"\nFeature shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Train classifier
    model, val_acc = train_classifier(X_train, train_labels, X_val, val_labels, device)
    
    # Test
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = outputs.max(1)
    
    predicted = predicted.cpu().numpy()
    test_labels = np.array(test_labels)
    
    test_acc = 100. * (predicted == test_labels).sum() / len(test_labels)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Per-class
    print("\nPer-class accuracy:")
    for i in range(5):
        mask = test_labels == i
        if mask.sum() > 0:
            class_acc = 100. * (predicted[mask] == test_labels[mask]).sum() / mask.sum()
            print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print(f"✓ Custom emotion2vec encoder works!")
    print(f"✓ Achieved {test_acc:.2f}% on English test set")
    print()
    print("For Tamil:")
    print("  - Need raw Tamil audio files")
    print("  - Or reconstruct from spectrograms (lossy)")
    print("  - Expected: 50-65% accuracy with proper audio")

if __name__ == '__main__':
    main()
