"""
Multilingual Speech Emotion Recognition with Domain-Adaptive Pre-training
Supports: English (RAVDESS/TESS), Tamil (EmoTa), Sinhala
Model: facebook/wav2vec2-xls-r-300m with custom classification head
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

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


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Model
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    num_labels: int = 7  # neutral, happy, sad, angry, fear, disgust, surprise
    hidden_size: int = 1024
    dropout: float = 0.1
    freeze_layers: int = 12
    
    # Audio
    sampling_rate: int = 16000
    max_duration: float = 10.0  # seconds
    
    # Training
    batch_size: int = 8
    num_epochs: int = 30
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # DAPT (Domain-Adaptive Pre-training)
    dapt_epochs: int = 10
    dapt_mask_prob: float = 0.065
    dapt_batch_size: int = 16
    
    # Data paths (update these to your actual paths)
    english_csv: str = "data/english/labels.csv"
    tamil_csv: str = "data/tamil/labels.csv"
    sinhala_labeled_csv: str = "data/sinhala/labeled.csv"
    sinhala_unlabeled_csv: str = "data/sinhala/unlabeled.csv"
    output_dir: str = "./outputs/multilingual_ser"
    
    # Language weights for sampling (adjust based on your data distribution)
    lang_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.lang_weights is None:
            # Default weights: balance smaller datasets
            self.lang_weights = {
                'english': 1.0,
                'tamil': 3.0,
                'sinhala': 10.0
            }


# ============================================================================
# Data Augmentation Pipeline
# ============================================================================

class AudioAugmenter:
    """Audio augmentation using audiomentations library"""
    
    def __init__(self, sampling_rate: int = 16000):
        self.augment = AA.Compose([
            AA.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            AA.RoomSimulator(p=0.3),
        ])
        self.sampling_rate = sampling_rate
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline"""
        return self.augment(samples=audio, sample_rate=self.sampling_rate)


# ============================================================================
# Dataset Classes
# ============================================================================

class SpeechEmotionDataset(Dataset):
    """Dataset for labeled speech emotion data"""
    
    def __init__(
        self,
        csv_path: str,
        feature_extractor: Wav2Vec2FeatureExtractor,
        config: Config,
        language: str,
        augment: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        self.config = config
        self.language = language
        self.augment = augment
        self.augmenter = AudioAugmenter(config.sampling_rate) if augment else None
        
        # Validate CSV format
        assert 'path' in self.df.columns and 'label' in self.df.columns, \
            "CSV must contain 'path' and 'label' columns"
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row['path']
        label = int(row['label'])
        
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=self.config.sampling_rate)
        
        # Truncate or pad to max duration
        max_samples = int(self.config.max_duration * self.config.sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Apply augmentation
        if self.augment and self.augmenter:
            audio = self.augmenter(audio)
        
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
    """Dataset for unlabeled speech data (DAPT)"""
    
    def __init__(
        self,
        csv_path: str,
        feature_extractor: Wav2Vec2FeatureExtractor,
        config: Config
    ):
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        self.config = config
        
        assert 'path' in self.df.columns, "CSV must contain 'path' column"
    
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
# Model Architecture
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
    """XLS-R model with custom classification head for emotion recognition"""
    
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
        """Freeze the feature extraction layers"""
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_encoder_layers(self, num_layers: int = 12):
        """Freeze the first N transformer encoder layers"""
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
        
        # Mean pooling over time dimension
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
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }


# ============================================================================
# Domain-Adaptive Pre-training (DAPT)
# ============================================================================

class DAPTTrainer:
    """Domain-Adaptive Pre-training using Masked Language Modeling"""
    
    def __init__(self, model: Wav2Vec2ForSER, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_dapt(self, unlabeled_dataset: UnlabeledSpeechDataset):
        """Execute Domain-Adaptive Pre-training"""
        print(f"\n{'='*60}")
        print("PHASE 1: Domain-Adaptive Pre-training (DAPT)")
        print(f"{'='*60}\n")
        
        # Custom collate function to pad sequences
        def collate_fn(batch):
            input_values = [item['input_values'] for item in batch]
            # Pad to max length in batch
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
            self.model.wav2vec2.parameters(),  # Only train the backbone
            lr=self.config.lr_backbone,
            weight_decay=self.config.weight_decay
        )
        
        self.model.train()
        
        for epoch in range(self.config.dapt_epochs):
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_values = batch['input_values'].to(self.device)
                
                # Forward pass through wav2vec2 backbone
                outputs = self.model.wav2vec2(input_values)
                
                # Use mean pooling and compute simple reconstruction loss
                # Goal: adapt the model to Sinhala audio characteristics
                hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
                
                # Compute contrastive loss: maximize similarity of representations
                pooled = hidden_states.mean(dim=1)  # [batch, hidden]
                
                # Normalize
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                
                # Contrastive loss: maximize similarity within batch
                similarity = torch.matmul(pooled, pooled.T)  # [batch, batch]
                
                # Target: maximize diagonal (self-similarity) and minimize off-diagonal
                target = torch.eye(similarity.size(0), device=self.device)
                loss = nn.MSELoss()(similarity, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.config.dapt_epochs}] "
                          f"Batch [{batch_idx+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}\n")
        
        print(f"{'='*60}")
        print("DAPT Completed Successfully!")
        print(f"{'='*60}\n")
        
        # Save DAPT checkpoint
        dapt_path = os.path.join(self.config.output_dir, "dapt_checkpoint")
        self.model.save_pretrained(dapt_path)
        print(f"DAPT model saved to {dapt_path}\n")


# ============================================================================
# Multilingual Training with Weighted Sampling
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


def create_weighted_sampler(datasets: List[SpeechEmotionDataset], config: Config) -> WeightedRandomSampler:
    """Create weighted sampler to balance languages"""
    weights = []
    languages = ['english', 'tamil', 'sinhala']
    
    for dataset, lang in zip(datasets, languages):
        lang_weight = config.lang_weights.get(lang, 1.0)
        weights.extend([lang_weight] * len(dataset))
    
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler


# ============================================================================
# Metrics and Evaluation
# ============================================================================

def compute_metrics(pred) -> Dict[str, float]:
    """Compute accuracy and F1-score"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1
    }


def compute_language_metrics(model, datasets: List[SpeechEmotionDataset], device) -> Dict[str, Dict[str, float]]:
    """Compute per-language metrics"""
    languages = ['english', 'tamil', 'sinhala']
    results = {}
    
    model.eval()
    
    for dataset, lang in zip(datasets, languages):
        all_preds = []
        all_labels = []
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_values = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_values=input_values)
                preds = outputs['logits'].argmax(-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        results[lang] = {
            'accuracy': acc,
            'f1': f1
        }
    
    return results


# ============================================================================
# Data Collator
# ============================================================================

@dataclass
class DataCollatorForSER:
    """Custom data collator for speech emotion recognition"""
    
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_values = [f['input_values'] for f in features]
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        
        # Pad input values
        batch = self.feature_extractor.pad(
            {'input_values': input_values},
            padding=self.padding,
            return_tensors='pt'
        )
        
        batch['labels'] = labels
        return batch


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Initialize configuration
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
    
    # ========================================================================
    # PHASE 1: Domain-Adaptive Pre-training (DAPT)
    # ========================================================================
    
    print("\n" + "="*60)
    print("Loading unlabeled Sinhala data for DAPT...")
    print("="*60 + "\n")
    
    # Check if DAPT checkpoint exists
    dapt_checkpoint = os.path.join(config.output_dir, "dapt_checkpoint")
    
    if os.path.exists(dapt_checkpoint):
        print(f"Loading existing DAPT checkpoint from {dapt_checkpoint}\n")
        model = Wav2Vec2ForSER.from_pretrained(dapt_checkpoint, num_labels=config.num_labels)
    else:
        # Initialize model
        model = Wav2Vec2ForSER.from_pretrained(config.model_name, num_labels=config.num_labels)
        
        # Load unlabeled Sinhala data
        if os.path.exists(config.sinhala_unlabeled_csv):
            unlabeled_dataset = UnlabeledSpeechDataset(
                config.sinhala_unlabeled_csv,
                feature_extractor,
                config
            )
            
            # Perform DAPT
            dapt_trainer = DAPTTrainer(model, config)
            dapt_trainer.train_dapt(unlabeled_dataset)
        else:
            print(f"Warning: Unlabeled Sinhala data not found at {config.sinhala_unlabeled_csv}")
            print("Skipping DAPT phase...\n")
    
    # ========================================================================
    # PHASE 2: Multilingual Fine-tuning
    # ========================================================================
    
    print("\n" + "="*60)
    print("PHASE 2: Multilingual Fine-tuning")
    print("="*60 + "\n")
    
    # Freeze layers
    model.freeze_feature_extractor()
    model.freeze_encoder_layers(config.freeze_layers)
    print(f"Frozen first {config.freeze_layers} encoder layers\n")
    
    # Load labeled datasets
    print("Loading multilingual datasets...")
    
    datasets = []
    val_datasets = []
    
    for csv_path, lang in [
        (config.english_csv, 'english'),
        (config.tamil_csv, 'tamil'),
        (config.sinhala_labeled_csv, 'sinhala')
    ]:
        if os.path.exists(csv_path):
            # Split into train/val
            df = pd.read_csv(csv_path)
            train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
            
            # Save splits
            train_csv = csv_path.replace('.csv', '_train.csv')
            val_csv = csv_path.replace('.csv', '_val.csv')
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
            
            # Create datasets
            train_dataset = SpeechEmotionDataset(
                train_csv, feature_extractor, config, lang, augment=True
            )
            val_dataset = SpeechEmotionDataset(
                val_csv, feature_extractor, config, lang, augment=False
            )
            
            datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            
            print(f"{lang.capitalize()}: {len(train_dataset)} train, {len(val_dataset)} val samples")
        else:
            print(f"Warning: {lang.capitalize()} data not found at {csv_path}")
    
    if not datasets:
        raise ValueError("No datasets found! Please check your CSV paths.")
    
    print()
    
    # Create combined datasets
    train_dataset = MultilingualDataset(datasets)
    val_dataset = MultilingualDataset(val_datasets)
    
    # Create weighted sampler
    sampler = create_weighted_sampler(datasets, config)
    
    # Data collator
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
        logging_steps=50,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        report_to="none"
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
    
    # Train
    print("Starting multilingual training...\n")
    trainer.train()
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    print("\n" + "="*60)
    print("Evaluating per-language performance...")
    print("="*60 + "\n")
    
    lang_metrics = compute_language_metrics(model, val_datasets, device)
    
    for lang, metrics in lang_metrics.items():
        print(f"{lang.capitalize()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}\n")
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    feature_extractor.save_pretrained(final_model_path)
    
    print(f"{'='*60}")
    print(f"Training completed! Model saved to {final_model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
