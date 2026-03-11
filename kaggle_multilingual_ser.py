"""
===============================================================================
MULTILINGUAL SPEECH EMOTION RECOGNITION WITH DAPT - KAGGLE VERSION
===============================================================================

Instructions for Kaggle:
1. Create a new Kaggle Notebook
2. Enable GPU: Settings → Accelerator → GPU T4 x2
3. Add datasets:
   - RAVDESS: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
   - TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
   - Tamil (Optional): Upload your EmoTa dataset or use a subset of English
4. Copy this entire script into a code cell and run
5. Training will take ~4-6 hours on Kaggle GPU

Model: facebook/wav2vec2-xls-r-300m
Strategy: Domain-Adaptive Pre-training (DAPT) + Multilingual Fine-tuning
Languages: English, Tamil, Sinhala (mock)
===============================================================================
"""

# ============================================================================
# SECTION 1: INSTALLATION & IMPORTS
# ============================================================================

# Install required packages
import subprocess
import sys

print("="*80)
print("Installing required packages...")
print("="*80)

packages = [
    'audiomentations',
    'transformers',
    'datasets',
    'accelerate',
    'scikit-learn',
    'pandas',
    'numpy',
    'librosa',
    'soundfile'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("✓ All packages installed successfully!\n")

# Standard imports
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
import audiomentations as AA
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

print("✓ All imports successful!\n")


# ============================================================================
# SECTION 2: KAGGLE DATASET PREPARATION
# ============================================================================

print("="*80)
print("Preparing Kaggle datasets...")
print("="*80)

# Kaggle dataset paths (update these based on your added datasets)
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Common Kaggle dataset names
RAVDESS_PATH = None
TESS_PATH = None

# Auto-detect Kaggle datasets
for item in os.listdir(KAGGLE_INPUT):
    item_path = os.path.join(KAGGLE_INPUT, item)
    if 'ravdess' in item.lower():
        RAVDESS_PATH = item_path
    elif 'tess' in item.lower() or 'toronto' in item.lower():
        TESS_PATH = item_path

print(f"RAVDESS detected at: {RAVDESS_PATH}")
print(f"TESS detected at: {TESS_PATH}")
print()


def find_audio_files(base_path, extension=".wav"):
    """Recursively find all audio files"""
    audio_files = []
    if base_path and os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(extension):
                    audio_files.append(os.path.join(root, file))
    return audio_files


def prepare_ravdess_data(base_path):
    """Prepare RAVDESS dataset"""
    print("Processing RAVDESS dataset...")
    data = []
    
    if not base_path:
        print("  Warning: RAVDESS not found")
        return pd.DataFrame(columns=['path', 'label'])
    
    # RAVDESS emotion mapping: 01=neutral, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
    emotion_map = {'01': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
    
    audio_files = find_audio_files(base_path)
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        parts = filename.split('-')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                data.append({
                    'path': audio_file,
                    'label': emotion_map[emotion_code]
                })
    
    print(f"  ✓ Found {len(data)} RAVDESS samples")
    return pd.DataFrame(data)


def prepare_tess_data(base_path):
    """Prepare TESS dataset"""
    print("Processing TESS dataset...")
    data = []
    
    if not base_path:
        print("  Warning: TESS not found")
        return pd.DataFrame(columns=['path', 'label'])
    
    # TESS emotion mapping
    emotion_keywords = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'disgust': 5,
        'ps': 6  # pleasant surprise
    }
    
    audio_files = find_audio_files(base_path)
    
    for audio_file in audio_files:
        path_lower = audio_file.lower()
        
        for keyword, label in emotion_keywords.items():
            if keyword in path_lower:
                data.append({
                    'path': audio_file,
                    'label': label
                })
                break
    
    print(f"  ✓ Found {len(data)} TESS samples")
    return pd.DataFrame(data)


# Prepare datasets
ravdess_df = prepare_ravdess_data(RAVDESS_PATH)
tess_df = prepare_tess_data(TESS_PATH)

# Combine English datasets
english_df = pd.concat([ravdess_df, tess_df], ignore_index=True)
english_df = english_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create Tamil dataset (use subset of English as placeholder if no Tamil data)
tamil_df = english_df.sample(min(936, len(english_df)//5), random_state=42).reset_index(drop=True)

# Create Sinhala datasets (mock data for demonstration)
sinhala_labeled_df = english_df.sample(100, random_state=42).reset_index(drop=True)
sinhala_unlabeled_df = english_df.sample(500, random_state=43)[['path']].reset_index(drop=True)

print(f"\n{'='*80}")
print("Dataset Summary:")
print(f"  English: {len(english_df)} samples")
print(f"  Tamil: {len(tamil_df)} samples")
print(f"  Sinhala: {len(sinhala_labeled_df)} labeled + {len(sinhala_unlabeled_df)} unlabeled")
print(f"{'='*80}\n")


# ============================================================================
# SECTION 3: CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    num_labels: int = 7
    hidden_size: int = 1024
    dropout: float = 0.1
    freeze_layers: int = 12
    
    # Audio
    sampling_rate: int = 16000
    max_duration: float = 10.0
    
    # Training
    batch_size: int = 16  # Kaggle GPU can handle this
    num_epochs: int = 20
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # DAPT
    dapt_epochs: int = 5
    dapt_batch_size: int = 16
    
    # Output
    output_dir: str = "/kaggle/working/multilingual_ser"
    
    # Language weights
    lang_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.lang_weights is None:
            self.lang_weights = {
                'english': 1.0,
                'tamil': 3.0,
                'sinhala': 10.0
            }

config = Config()
os.makedirs(config.output_dir, exist_ok=True)


# ============================================================================
# SECTION 4: DATA AUGMENTATION
# ============================================================================

class AudioAugmenter:
    """Audio augmentation pipeline"""
    
    def __init__(self, sampling_rate: int = 16000):
        self.augment = AA.Compose([
            AA.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            AA.RoomSimulator(p=0.3),
        ])
        self.sampling_rate = sampling_rate
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.augment(samples=audio, sample_rate=self.sampling_rate)


# ============================================================================
# SECTION 5: DATASET CLASSES
# ============================================================================

class SpeechEmotionDataset(Dataset):
    """Dataset for labeled speech emotion data"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Wav2Vec2FeatureExtractor,
        config: Config,
        language: str,
        augment: bool = False
    ):
        self.df = df
        self.feature_extractor = feature_extractor
        self.config = config
        self.language = language
        self.augment = augment
        self.augmenter = AudioAugmenter(config.sampling_rate) if augment else None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row['path']
        label = int(row['label'])
        
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=self.config.sampling_rate)
        
        # Truncate or pad
        max_samples = int(self.config.max_duration * self.config.sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Apply augmentation
        if self.augment and self.augmenter:
            try:
                audio = self.augmenter(audio)
            except:
                pass  # Skip augmentation if it fails
        
        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=max_samples,
            truncation=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'language': self.language
        }


class UnlabeledSpeechDataset(Dataset):
    """Dataset for unlabeled speech (DAPT)"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Wav2Vec2FeatureExtractor,
        config: Config
    ):
        self.df = df
        self.feature_extractor = feature_extractor
        self.config = config
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.df.iloc[idx]['path']
        
        audio, sr = librosa.load(audio_path, sr=self.config.sampling_rate)
        
        max_samples = int(self.config.max_duration * self.config.sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=max_samples,
            truncation=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0)
        }


# ============================================================================
# SECTION 6: MODEL ARCHITECTURE
# ============================================================================

class ClassificationHead(nn.Module):
    """Custom classification head: Linear -> Dropout -> Tanh -> Linear"""
    
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dense(features)
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSER(Wav2Vec2PreTrainedModel):
    """XLS-R model with custom classification head"""
    
    def __init__(self, config, num_labels: int = 7):
        super().__init__(config)
        self.num_labels = num_labels
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = ClassificationHead(
            hidden_size=config.hidden_size,
            num_labels=num_labels,
            dropout=0.1
        )
        self.init_weights()
    
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_encoder_layers(self, num_layers: int = 12):
        for layer in self.wav2vec2.encoder.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )
        
        # Mean pooling
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states * attention_mask
            pooled_output = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


# ============================================================================
# SECTION 7: DOMAIN-ADAPTIVE PRE-TRAINING (DAPT)
# ============================================================================

class DAPTTrainer:
    """Domain-Adaptive Pre-training"""
    
    def __init__(self, model: Wav2Vec2ForSER, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_dapt(self, unlabeled_dataset: UnlabeledSpeechDataset):
        print(f"\n{'='*80}")
        print("PHASE 1: Domain-Adaptive Pre-training (DAPT)")
        print(f"{'='*80}\n")
        
        def collate_fn(batch):
            input_values = [item['input_values'] for item in batch]
            max_len = max(len(x) for x in input_values)
            padded = torch.zeros(len(input_values), max_len)
            for i, x in enumerate(input_values):
                padded[i, :len(x)] = x
            return {'input_values': padded}
        
        dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=self.config.dapt_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        optimizer = torch.optim.AdamW(
            self.model.wav2vec2.parameters(),
            lr=self.config.lr_backbone,
            weight_decay=self.config.weight_decay
        )
        
        self.model.train()
        
        for epoch in range(self.config.dapt_epochs):
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_values = batch['input_values'].to(self.device)
                
                outputs = self.model.wav2vec2(input_values)
                hidden_states = outputs.last_hidden_state
                
                # Contrastive learning objective
                pooled = hidden_states.mean(dim=1)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                
                similarity = torch.matmul(pooled, pooled.T)
                target = torch.eye(similarity.size(0), device=self.device)
                loss = nn.MSELoss()(similarity, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{self.config.dapt_epochs}] "
                          f"Batch [{batch_idx+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}\n")
        
        print(f"{'='*80}")
        print("DAPT Completed!")
        print(f"{'='*80}\n")
        
        dapt_path = os.path.join(self.config.output_dir, "dapt_checkpoint")
        self.model.save_pretrained(dapt_path)
        print(f"✓ DAPT model saved to {dapt_path}\n")


# ============================================================================
# SECTION 8: MULTILINGUAL TRAINING UTILITIES
# ============================================================================

class MultilingualDataset(Dataset):
    """Combined dataset for multilingual training"""
    
    def __init__(self, datasets: List[SpeechEmotionDataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
    
    def __len__(self) -> int:
        return sum(self.lengths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


def create_weighted_sampler(datasets: List[SpeechEmotionDataset], config: Config):
    """Create weighted sampler for language balancing"""
    weights = []
    languages = ['english', 'tamil', 'sinhala']
    
    for dataset, lang in zip(datasets, languages):
        lang_weight = config.lang_weights.get(lang, 1.0)
        weights.extend([lang_weight] * len(dataset))
    
    weights = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


@dataclass
class DataCollatorForSER:
    """Data collator for speech emotion recognition"""
    
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_values = [f['input_values'] for f in features]
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        
        batch = self.feature_extractor.pad(
            {'input_values': input_values},
            padding=self.padding,
            return_tensors='pt'
        )
        
        batch['labels'] = labels
        return batch


def compute_metrics(pred) -> Dict[str, float]:
    """Compute accuracy and F1-score"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    return {'accuracy': acc, 'f1': f1}


# ============================================================================
# SECTION 9: MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print(f"\n{'='*80}")
    print("MULTILINGUAL SPEECH EMOTION RECOGNITION - TRAINING")
    print(f"{'='*80}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Initialize feature extractor
    print("Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
    print("✓ Feature extractor loaded\n")
    
    # ========================================================================
    # PHASE 1: DAPT
    # ========================================================================
    
    dapt_checkpoint = os.path.join(config.output_dir, "dapt_checkpoint")
    
    if os.path.exists(dapt_checkpoint):
        print(f"Loading existing DAPT checkpoint from {dapt_checkpoint}\n")
        model = Wav2Vec2ForSER.from_pretrained(dapt_checkpoint, num_labels=config.num_labels)
    else:
        print("Initializing model...")
        model = Wav2Vec2ForSER.from_pretrained(config.model_name, num_labels=config.num_labels)
        print("✓ Model initialized\n")
        
        if len(sinhala_unlabeled_df) > 0:
            unlabeled_dataset = UnlabeledSpeechDataset(
                sinhala_unlabeled_df,
                feature_extractor,
                config
            )
            
            dapt_trainer = DAPTTrainer(model, config)
            dapt_trainer.train_dapt(unlabeled_dataset)
        else:
            print("Skipping DAPT (no unlabeled data)\n")
    
    # ========================================================================
    # PHASE 2: MULTILINGUAL FINE-TUNING
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("PHASE 2: Multilingual Fine-tuning")
    print(f"{'='*80}\n")
    
    # Freeze layers
    model.freeze_feature_extractor()
    model.freeze_encoder_layers(config.freeze_layers)
    print(f"✓ Frozen first {config.freeze_layers} encoder layers\n")
    
    # Create train/val splits
    print("Preparing datasets...")
    
    english_train, english_val = train_test_split(
        english_df, test_size=0.15, random_state=42, stratify=english_df['label']
    )
    tamil_train, tamil_val = train_test_split(
        tamil_df, test_size=0.15, random_state=42, stratify=tamil_df['label']
    )
    sinhala_train, sinhala_val = train_test_split(
        sinhala_labeled_df, test_size=0.15, random_state=42, stratify=sinhala_labeled_df['label']
    )
    
    # Create datasets
    train_datasets = [
        SpeechEmotionDataset(english_train, feature_extractor, config, 'english', augment=True),
        SpeechEmotionDataset(tamil_train, feature_extractor, config, 'tamil', augment=True),
        SpeechEmotionDataset(sinhala_train, feature_extractor, config, 'sinhala', augment=True)
    ]
    
    val_datasets = [
        SpeechEmotionDataset(english_val, feature_extractor, config, 'english', augment=False),
        SpeechEmotionDataset(tamil_val, feature_extractor, config, 'tamil', augment=False),
        SpeechEmotionDataset(sinhala_val, feature_extractor, config, 'sinhala', augment=False)
    ]
    
    print(f"English: {len(english_train)} train, {len(english_val)} val")
    print(f"Tamil: {len(tamil_train)} train, {len(tamil_val)} val")
    print(f"Sinhala: {len(sinhala_train)} train, {len(sinhala_val)} val")
    print()
    
    train_dataset = MultilingualDataset(train_datasets)
    val_dataset = MultilingualDataset(val_datasets)
    
    sampler = create_weighted_sampler(train_datasets, config)
    data_collator = DataCollatorForSER(feature_extractor=feature_extractor)
    
    # Differential learning rates
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if 'wav2vec2' in n and p.requires_grad],
            'lr': config.lr_backbone
        },
        {
            'params': [p for n, p in model.named_parameters() if 'classifier' in n],
            'lr': config.lr_head
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=False,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Override sampler
    trainer._get_train_sampler = lambda: sampler
    
    print("Starting multilingual training...\n")
    print("="*80)
    
    # Train
    trainer.train()
    
    # ========================================================================
    # EVALUATION & SAVE
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("Training Complete! Saving model...")
    print(f"{'='*80}\n")
    
    final_model_path = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    feature_extractor.save_pretrained(final_model_path)
    
    print(f"✓ Model saved to {final_model_path}")
    print(f"\n{'='*80}")
    print("ALL DONE! 🎉")
    print(f"{'='*80}\n")
    
    print("Next steps:")
    print("1. Download the model from /kaggle/working/multilingual_ser/final_model/")
    print("2. Use for inference on new audio files")
    print("3. Fine-tune further on your specific dataset")
    print()


# ============================================================================
# SECTION 10: EXECUTE TRAINING
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║         MULTILINGUAL SPEECH EMOTION RECOGNITION WITH DAPT                ║
    ║                                                                          ║
    ║  Model: facebook/wav2vec2-xls-r-300m (300M parameters)                  ║
    ║  Strategy: Domain-Adaptive Pre-training + Multilingual Fine-tuning      ║
    ║  Languages: English, Tamil, Sinhala                                     ║
    ║                                                                          ║
    ║  Estimated Time: 4-6 hours on Kaggle GPU T4                             ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    main()
