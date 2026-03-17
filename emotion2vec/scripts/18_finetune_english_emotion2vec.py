"""
Fine-tune emotion2vec model on English audio (end-to-end)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import time

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}
TARGET_SR = 16000
TARGET_LENGTH = 48000

# ===== Model Classes =====

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
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        
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
        features = self.encoder(audio)
        features = features.mean(dim=1)
        output = self.classifier(features)
        return output

# ===== Dataset =====

class EnglishAudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=TARGET_SR)
        
        # Pad or trim
        if len(audio) < TARGET_LENGTH:
            audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
        else:
            audio = audio[:TARGET_LENGTH]
        
        return torch.FloatTensor(audio), self.labels[idx]

def collect_english_files():
    """Collect all English audio files"""
    data_root = Path('../data/raw')
    files = []
    labels = []
    
    # RAVDESS
    ravdess_path = data_root / 'RAVDESS-SPEECH'
    if ravdess_path.exists():
        for audio_file in ravdess_path.rglob('*.wav'):
            parts = audio_file.stem.split('-')
            emotion_code = int(parts[2])
            emotion_map = {3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 1: 'neutral'}
            if emotion_code in emotion_map:
                files.append(str(audio_file))
                labels.append(EMOTION_TO_INDEX[emotion_map[emotion_code]])
    
    # TESS
    tess_path = data_root / 'TESS'
    if tess_path.exists():
        for audio_file in tess_path.rglob('*.wav'):
            filename = audio_file.stem.lower()
            if 'angry' in filename:
                emotion = 'angry'
            elif 'fear' in filename:
                emotion = 'fear'
            elif 'happy' in filename or 'ps' in filename:
                emotion = 'happy'
            elif 'neutral' in filename:
                emotion = 'neutral'
            elif 'sad' in filename:
                emotion = 'sad'
            else:
                continue
            files.append(str(audio_file))
            labels.append(EMOTION_TO_INDEX[emotion])
    
    return files, labels

# ===== Training =====

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for encoder and classifier
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 0.00001},  # Low LR for pretrained
        {'params': model.classifier.parameters(), 'lr': 0.001}   # Higher LR for new head
    ])
    
    best_val_acc = 0
    patience = 0
    max_patience = 10
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for audio, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train={train_acc:.2f}% Val={val_acc:.2f}%")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_english_finetuned.pt')
            patience = 0
            print(f"  → New best! Saved.")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    return best_val_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, labels in tqdm(test_loader, desc="Testing"):
            audio = audio.to(device)
            outputs = model(audio)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    print("\n" + "=" * 70)
    print("FINE-TUNED MODEL RESULTS")
    print("=" * 70)
    print(f"\nTest Accuracy: {accuracy:.2f}%\n")
    
    print("Per-class accuracy:")
    for i in range(5):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100. * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({mask.sum()} samples)")
    
    return accuracy

# ===== Main =====

def main():
    print("=" * 70)
    print("FINE-TUNING EMOTION2VEC ON ENGLISH")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("Collecting audio files...")
    files, labels = collect_english_files()
    print(f"Total: {len(files)} files\n")
    
    # Split
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(files))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    labels_temp = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_temp)
    
    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}\n")
    
    # Create datasets
    train_dataset = EnglishAudioDataset(train_files, train_labels)
    val_dataset = EnglishAudioDataset(val_files, val_labels)
    test_dataset = EnglishAudioDataset(test_files, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Load model
    print("Loading emotion2vec encoder...")
    checkpoint_path = '../emotion2vec_base/emotion2vec_base.pt'
    encoder = SimpleEmotion2VecEncoder(checkpoint_path)
    model = EmotionModel(encoder)
    model.to(device)
    print("✓ Model loaded\n")
    
    # Train
    print("Starting fine-tuning...")
    print("This will take ~20-30 minutes on CPU\n")
    
    start_time = time.time()
    best_val_acc = train_model(model, train_loader, val_loader, device, num_epochs=50)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed/60:.1f} minutes\n")
    
    # Test
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_english_finetuned.pt'))
    test_acc = evaluate_model(model, test_loader, device)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"CNN (baseline):             94.89%")
    print(f"emotion2vec (frozen):       31.65%")
    print(f"emotion2vec (fine-tuned):   {test_acc:.2f}%")
    
    if test_acc > 90:
        print("\n🎉 EXCELLENT! emotion2vec fine-tuning works great on English!")
        print("    This validates the approach for transfer learning.")
    elif test_acc > 80:
        print("\n✓ Good! emotion2vec performs reasonably but CNN is better.")
    elif test_acc > 50:
        print("\n✓ Fine-tuning helps but doesn't reach CNN performance.")
    else:
        print("\n⚠ Fine-tuning didn't help much. emotion2vec may not be suitable.")

if __name__ == '__main__':
    main()
