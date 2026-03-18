"""
Check the fine-tuned model results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

EMOTION_TO_INDEX = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
INDEX_TO_EMOTION = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

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

# ===== Evaluation =====

def evaluate_on_features():
    """Evaluate using pre-extracted features (faster)"""
    print("=" * 70)
    print("EVALUATING FINE-TUNED MODEL")
    print("=" * 70)
    print()
    
    # Load features
    feature_dir = '../features/tamil_emotion2vec'
    X_test = np.load(os.path.join(feature_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(feature_dir, 'y_test.npy'))
    
    print(f"Test samples: {len(X_test)}\n")
    
    # Load classifier (simplified - just the classifier head)
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(256),
                
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(128),
                
                nn.Linear(128, 5)
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    # Try to extract classifier weights from full model
    print("Loading fine-tuned model...")
    try:
        checkpoint_path = '../emotion2vec_base/emotion2vec_base.pt'
        encoder = SimpleEmotion2VecEncoder(checkpoint_path)
        full_model = EmotionModel(encoder)
        
        # Load fine-tuned weights
        full_model.load_state_dict(torch.load('best_tamil_finetuned.pt', map_location='cpu'))
        
        print("✓ Model loaded successfully\n")
        
        # Extract just the classifier
        classifier = full_model.classifier
        classifier.eval()
        
        # Evaluate
        X_test_tensor = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            outputs = classifier(X_test_tensor)
            _, predicted = outputs.max(1)
        
        predicted = predicted.numpy()
        
        # Calculate accuracy
        accuracy = 100. * (predicted == y_test).sum() / len(y_test)
        
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nTest Accuracy: {accuracy:.2f}%\n")
        
        # Per-class
        print("Per-class accuracy:")
        for i in range(5):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = 100. * (predicted[mask] == y_test[mask]).sum() / mask.sum()
                print(f"  {INDEX_TO_EMOTION[i]:8s}: {class_acc:6.2f}% ({mask.sum()} samples)")
        
        # Comparison
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"Simple CNN:                34.04%")
        print(f"Feature-based (augmented): 36.88%")
        print(f"emotion2vec (frozen):      29.79%")
        print(f"emotion2vec (fine-tuned):  {accuracy:.2f}%")
        
        improvement = accuracy - 36.88
        print(f"\nImprovement over previous: {improvement:+.2f}%")
        
        if accuracy > 50:
            print("\n🎉 EXCELLENT! Fine-tuning dramatically improved results!")
        elif accuracy > 45:
            print("\n✓ Great! Fine-tuning significantly helped")
        elif accuracy > 40:
            print("\n✓ Good! Fine-tuning provided improvement")
        elif accuracy > 36.88:
            print("\n✓ Minor improvement with fine-tuning")
        else:
            print("\n⚠ Fine-tuning didn't improve over feature-based approach")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    evaluate_on_features()
