"""
Fine-tune emotion2vec encoder on Tamil data
Instead of freezing the encoder, we'll train it end-to-end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import soundfile as sf
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

# ===== Custom Encoder =====

class ConvFeatureExtraction(nn.Module):
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
            x = F.gelu(x)
        return x

class SimpleEmotion2VecEncoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.state_dict_full = checkpoint['model']
        
        conv_configs = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 3, 2), (512, 2, 2),
        ]
        
        self.feature_extractor = ConvFeatureExtraction(conv_configs)
        self._load_conv_weights()
        self.post_extract_proj = nn.Linear(512, 768)
    
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
    
    def forward(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        features = self.feature_extractor(audio)
        features = features.transpose(1, 2)
        features = self.post_extract_proj(features)
        
        return features

class EmotionModel(nn.Module):
    """Complete model: Encoder + Classifier"""
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, audio):
        # Extract features from encoder
        features = self.encoder(audio)  # (batch, time, 768)
        
        # Global average pooling
        features = features.mean(dim=1)  # (batch, 768)
        
        # Classify
        output = self.classifier(features)
        
        return output

# ===== Dataset =====

class TamilAudioDataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, sr = sf.read(audio_file)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Pad or trim to 3 seconds
            target_length = 48000
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
        except Exception as e:
            audio = np.zeros(48000)
        
        return torch.FloatTensor(audio), label

def load_tamil_files():
    """Load Tamil audio file paths"""
    data_dir = '../../cnn/data/raw/EmoTa/TamilSER-DB'
    
    audio_files = []
    labels = []
    
    emotion_map = {
        'ang': 'angry', 'fea': 'fear', 'hap': 'happy',
        'neu': 'neutral', 'sad': 'sad'
    }
    
    for wav_file in glob.glob(os.path.join(data_dir, '*.wav')):
        filename = os.path.basename(wav_file)
        parts = filename.replace('.wav', '').split('_')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                audio_files.append(wav_file)
                labels.append(EMOTION_TO_INDEX[emotion])
    
    return audio_files, labels

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Train model with fine-tuning"""
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for encoder and classifier
    encoder_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 0.00001},  # Small LR for pretrained
        {'params': classifier_params, 'lr': 0.001}  # Normal LR for classifier
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print("\nFine-tuning model...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for audio, labels in train_loader:
            audio, labels = audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio)
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
            for audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tamil_finetuned.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load('best_tamil_finetuned.pt'))
    return model

def evaluate(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, labels in test_loader:
            audio = audio.to(device)
            outputs = model(audio)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    print("\nPer-class accuracy:")
    for i in range(5):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100. * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({mask.sum()} samples)")
    
    return accuracy

def main():
    print("=" * 70)
    print("FINE-TUNING EMOTION2VEC ON TAMIL DATA")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load encoder
    print("Loading pretrained encoder...")
    checkpoint_path = '../emotion2vec_base/emotion2vec_base.pt'
    encoder = SimpleEmotion2VecEncoder(checkpoint_path)
    
    # Create full model
    model = EmotionModel(encoder).to(device)
    print(f"✓ Model created")
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Load data
    audio_files, labels = load_tamil_files()
    print(f"Loaded {len(audio_files)} Tamil audio files\n")
    
    # Split
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        test_files, test_labels, test_size=0.5, stratify=test_labels, random_state=42
    )
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}\n")
    
    # Create datasets
    train_dataset = TamilAudioDataset(train_files, train_labels)
    val_dataset = TamilAudioDataset(val_files, val_labels)
    test_dataset = TamilAudioDataset(test_files, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0)
    
    # Train
    model = train_model(model, train_loader, val_loader, device, num_epochs=50)
    
    # Test
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Save
    torch.save(model.state_dict(), '../models/tamil_emotion2vec_finetuned.pt')
    print(f"\n✓ Model saved")
    
    # Compare
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Simple CNN:               34.04%")
    print(f"Feature-based (augmented): 36.88%")
    print(f"emotion2vec (frozen):     29.79%")
    print(f"emotion2vec (fine-tuned): {test_acc:.2f}%")
    
    if test_acc > 50:
        print("\n🎉 SUCCESS! Fine-tuning dramatically improved results!")
    elif test_acc > 40:
        print("\n✓ Good! Fine-tuning helped significantly")
    else:
        print("\n⚠ Fine-tuning didn't help as much as expected")

if __name__ == '__main__':
    main()
