"""
===============================================================================
MULTILINGUAL SPEECH EMOTION RECOGNITION WITH MMS-1B + LID
===============================================================================

Architecture:
1. Language Identification (LID) Layer - Detects language (English/Tamil/Sinhala)
2. Shared Encoder - Meta's MMS-1B (pre-trained multilingual speech model)
3. Language-Aware Classification Head - Conditioned on language code

Training Strategy:
- Phase A: Fine-tune on high-resource English (RAVDESS/TESS)
- Phase B: Joint training on all languages with higher sampling for Tamil/Sinhala

Model: facebook/mms-1b-all (1 billion parameters)
**REQUIRES GPU** - Kaggle T4 GPU recommended (16GB VRAM)

===============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings('ignore')

# Check for required packages
try:
    import audiomentations as AA
    from transformers import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
        Wav2Vec2FeatureExtractor,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install transformers datasets audiomentations scikit-learn librosa soundfile")
    sys.exit(1)


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model - Using MMS-300M (memory-efficient)
    model_name: str = "facebook/mms-300m"
    num_labels: int = 5  # 5 emotions: neutral, happy, sad, angry, fear
    num_languages: int = 3  # English, Tamil, Sinhala
    hidden_size: int = 1024  # MMS-300M hidden size
    dropout: float = 0.1
    freeze_layers: int = 15  # MMS-300M has ~24 layers
    
    # Language IDs
    lang_to_id: Dict[str, int] = None
    id_to_lang: Dict[int, str] = None
    
    # Audio
    sampling_rate: int = 16000
    max_duration: float = 10.0
    
    # Training - Phase A (English only)
    phase_a_epochs: int = 10
    phase_a_batch_size: int = 2  # Reduced for memory (MMS-1B)
    phase_a_lr_backbone: float = 5e-6
    phase_a_lr_head: float = 1e-4
    
    # Training - Phase B (Multilingual)
    phase_b_epochs: int = 15
    phase_b_batch_size: int = 2  # Reduced for memory
    phase_b_lr_backbone: float = 3e-6
    phase_b_lr_head: float = 5e-5
    
    # General training
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4  # Effective batch = 8 * 4 = 32
    
    # Language sampling weights for Phase B
    lang_weights: Dict[str, float] = None
    
    # Output
    output_dir: str = "./outputs/mms_multilingual_ser"
    
    def __post_init__(self):
        if self.lang_to_id is None:
            self.lang_to_id = {'english': 0, 'tamil': 1, 'sinhala': 2}
            self.id_to_lang = {v: k for k, v in self.lang_to_id.items()}
        
        if self.lang_weights is None:
            # Higher weights for low-resource languages
            self.lang_weights = {
                'english': 1.0,
                'tamil': 3.0,    # 3x more frequent sampling
                'sinhala': 5.0   # 5x more frequent sampling
            }

config = Config()
os.makedirs(config.output_dir, exist_ok=True)


# ============================================================================
# SECTION 2: LANGUAGE IDENTIFICATION (LID) LAYER
# ============================================================================

class LanguageIdentificationLayer(nn.Module):
    """
    Lightweight LID layer that predicts language from audio features.
    Acts as a gating mechanism to condition the emotion classifier.
    """
    
    def __init__(self, hidden_size: int, num_languages: int, dropout: float = 0.1):
        super().__init__()
        self.num_languages = num_languages
        
        # Lightweight classifier for language prediction
        self.lid_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_languages)
        )
    
    def forward(self, pooled_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled_output: [batch_size, hidden_size]
        
        Returns:
            lang_logits: [batch_size, num_languages] - raw language prediction scores
            lang_probs: [batch_size, num_languages] - softmax probabilities
        """
        lang_logits = self.lid_head(pooled_output)
        lang_probs = F.softmax(lang_logits, dim=-1)
        return lang_logits, lang_probs


# ============================================================================
# SECTION 3: LANGUAGE-AWARE CLASSIFICATION HEAD
# ============================================================================

class LanguageAwareEmotionHead(nn.Module):
    """
    Classification head that is conditioned on language code.
    Uses language embeddings to modulate emotion predictions.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        num_languages: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_labels = num_labels
        self.num_languages = num_languages
        
        # Language embeddings - learnable language-specific parameters
        self.lang_embeddings = nn.Embedding(num_languages, hidden_size // 4)
        
        # Emotion classifier with language conditioning
        self.pre_classifier = nn.Linear(hidden_size + hidden_size // 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(
        self,
        pooled_output: torch.Tensor,
        lang_probs: torch.Tensor,
        lang_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size] - audio features
            lang_probs: [batch_size, num_languages] - language probabilities from LID
            lang_ids: [batch_size] - ground truth language IDs (optional, for training)
        
        Returns:
            logits: [batch_size, num_labels] - emotion prediction logits
        """
        batch_size = pooled_output.size(0)
        
        # Use ground truth language IDs if available, otherwise use predicted
        if lang_ids is not None:
            # Training: use ground truth
            lang_embeds = self.lang_embeddings(lang_ids)  # [batch_size, hidden_size//4]
        else:
            # Inference: use soft attention over language embeddings
            # lang_probs: [batch_size, num_languages]
            all_lang_embeds = self.lang_embeddings.weight  # [num_languages, hidden_size//4]
            # Weighted sum: [batch_size, num_languages] @ [num_languages, hidden_size//4]
            lang_embeds = torch.matmul(lang_probs, all_lang_embeds)  # [batch_size, hidden_size//4]
        
        # Concatenate audio features with language embeddings
        combined = torch.cat([pooled_output, lang_embeds], dim=-1)  # [batch_size, hidden_size + hidden_size//4]
        
        # Emotion classification
        x = self.pre_classifier(combined)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits


# ============================================================================
# SECTION 4: MAIN MODEL ARCHITECTURE
# ============================================================================

class MMSForMultilingualSER(Wav2Vec2PreTrainedModel):
    """
    MMS-1B model with:
    1. Language Identification (LID) layer
    2. Shared encoder (MMS-1B)
    3. Language-aware emotion classification head
    """
    
    def __init__(self, config_obj: Config):
        # Initialize with MMS config
        from transformers import Wav2Vec2Config
        model_config = Wav2Vec2Config.from_pretrained(config_obj.model_name)
        super().__init__(model_config)
        
        self.config_obj = config_obj
        self.num_labels = config_obj.num_labels
        self.num_languages = config_obj.num_languages
        
        # Shared encoder: MMS-1B - load pretrained weights
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config_obj.model_name)
        
        # Language Identification layer
        self.lid_layer = LanguageIdentificationLayer(
            hidden_size=config_obj.hidden_size,
            num_languages=config_obj.num_languages,
            dropout=config_obj.dropout
        )
        
        # Language-aware emotion classification head
        self.emotion_head = LanguageAwareEmotionHead(
            hidden_size=config_obj.hidden_size,
            num_labels=config_obj.num_labels,
            num_languages=config_obj.num_languages,
            dropout=config_obj.dropout
        )
        
        # Initialize the custom layers only
        self._init_custom_layers()
    
    def _init_custom_layers(self):
        """Initialize weights for custom layers only"""
        # Initialize LID layer
        for module in self.lid_layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        
        # Initialize emotion head
        for module in self.emotion_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
    
    def freeze_feature_extractor(self):
        """Freeze the CNN feature extractor"""
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_encoder_layers(self, num_layers: int):
        """Freeze first N transformer encoder layers"""
        for layer in self.wav2vec2.encoder.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: [batch_size, seq_len] - audio waveform
            attention_mask: [batch_size, seq_len] - attention mask
            labels: [batch_size] - emotion labels
            lang_ids: [batch_size] - language IDs
        
        Returns:
            Dictionary with loss, emotion_logits, lang_logits, lang_probs
        """
        # 1. Extract audio features with shared encoder
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )
        
        # 2. Mean pooling over time dimension
        # Note: attention_mask is already handled internally by wav2vec2
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # 3. Language Identification
        lang_logits, lang_probs = self.lid_layer(pooled_output)
        
        # 4. Language-aware emotion classification
        emotion_logits = self.emotion_head(pooled_output, lang_probs, lang_ids)
        
        # 5. Compute losses
        total_loss = None
        emotion_loss = None
        lid_loss = None
        
        if labels is not None:
            # Emotion classification loss
            loss_fct = nn.CrossEntropyLoss()
            emotion_loss = loss_fct(emotion_logits, labels)
            total_loss = emotion_loss
            
            # Language identification loss (if lang_ids provided)
            if lang_ids is not None:
                lid_loss = loss_fct(lang_logits, lang_ids)
                # Combined loss: emotion is primary, LID is auxiliary
                total_loss = emotion_loss + 0.2 * lid_loss  # 0.2 weight for LID loss
        
        return {
            'loss': total_loss,
            'emotion_loss': emotion_loss,
            'lid_loss': lid_loss,
            'logits': (emotion_logits, lang_logits),  # Trainer expects 'logits' key
            'emotion_logits': emotion_logits,
            'lang_logits': lang_logits,
            'lang_probs': lang_probs
        }


# ============================================================================
# SECTION 5: DATA AUGMENTATION
# ============================================================================

class AudioAugmenter:
    """Audio augmentation for emotion data"""
    
    def __init__(self, sampling_rate: int = 16000):
        self.augment = AA.Compose([
            AA.PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
            AA.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
            AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
        ])
        self.sampling_rate = sampling_rate
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        try:
            return self.augment(samples=audio, sample_rate=self.sampling_rate)
        except:
            return audio  # Return original if augmentation fails


# ============================================================================
# SECTION 6: DATASET CLASS
# ============================================================================

class MultilingualEmotionDataset(Dataset):
    """Dataset for multilingual emotion recognition with language labels"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        processor: Wav2Vec2FeatureExtractor,
        config: Config,
        language: str,
        augment: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.config = config
        self.language = language
        self.lang_id = config.lang_to_id[language]
        self.augment = augment
        self.augmenter = AudioAugmenter(config.sampling_rate) if augment else None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row['path']
        label = int(row['label'])
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config.sampling_rate)
        
        # Truncate or pad
        max_samples = int(self.config.max_duration * self.config.sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Apply augmentation
        if self.augment and self.augmenter:
            audio = self.augmenter(audio)
        
        # Process audio
        inputs = self.processor(
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
            'lang_ids': torch.tensor(self.lang_id, dtype=torch.long)
        }


# ============================================================================
# SECTION 7: DATA UTILITIES
# ============================================================================

class CombinedDataset(Dataset):
    """Combines multiple language datasets"""
    
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
    
    def __len__(self) -> int:
        return sum(self.lengths)
    
    def __getitem__(self, idx: int) -> Dict:
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


def create_weighted_sampler(datasets: List[Dataset], config: Config) -> WeightedRandomSampler:
    """Create weighted sampler for balanced language sampling"""
    weights = []
    languages = ['english', 'tamil', 'sinhala']
    
    for dataset, lang in zip(datasets, languages):
        lang_weight = config.lang_weights.get(lang, 1.0)
        weights.extend([lang_weight] * len(dataset))
    
    weights = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


@dataclass
class DataCollatorForMultilingualSER:
    """Data collator with language IDs"""
    
    processor: Wav2Vec2FeatureExtractor
    padding: bool = True
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_values = [f['input_values'] for f in features]
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        lang_ids = torch.tensor([f['lang_ids'] for f in features], dtype=torch.long)
        
        batch = self.processor.pad(
            {'input_values': input_values},
            padding=self.padding,
            return_tensors='pt'
        )
        
        batch['labels'] = labels
        batch['lang_ids'] = lang_ids
        return batch


def compute_metrics(pred) -> Dict[str, float]:
    """Compute evaluation metrics"""
    labels = pred.label_ids
    
    # predictions is a tuple: (emotion_logits, lang_logits)
    # We only need emotion_logits for metrics
    if isinstance(pred.predictions, tuple):
        emotion_logits = pred.predictions[0]  # First element is emotion_logits
    else:
        emotion_logits = pred.predictions
    
    preds = emotion_logits.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    return {'accuracy': acc, 'f1': f1}


# ============================================================================
# SECTION 8: TRAINING FUNCTIONS
# ============================================================================

def phase_a_training(
    model: MMSForMultilingualSER,
    english_train_df: pd.DataFrame,
    english_val_df: pd.DataFrame,
    processor: Wav2Vec2FeatureExtractor,
    config: Config
):
    """
    Phase A: Fine-tune on high-resource English data
    Goal: Learn general emotional cues
    """
    print("\n" + "="*80)
    print("PHASE A: ENGLISH-ONLY FINE-TUNING")
    print("="*80 + "\n")
    
    # Create English datasets
    train_dataset = MultilingualEmotionDataset(
        english_train_df, processor, config, 'english', augment=True
    )
    val_dataset = MultilingualEmotionDataset(
        english_val_df, processor, config, 'english', augment=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Freeze early layers
    model.freeze_feature_extractor()
    model.freeze_encoder_layers(config.freeze_layers)
    print(f"Frozen first {config.freeze_layers} encoder layers\n")
    
    # Data collator
    data_collator = DataCollatorForMultilingualSER(processor=processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "phase_a"),
        per_device_train_batch_size=config.phase_a_batch_size,
        per_device_eval_batch_size=config.phase_a_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.phase_a_epochs,
        learning_rate=config.phase_a_lr_head,  # Will override with optimizer
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False
    )
    
    # Differential learning rates
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'wav2vec2' in n and p.requires_grad],
            'lr': config.phase_a_lr_backbone
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(x in n for x in ['lid_layer', 'emotion_head'])],
            'lr': config.phase_a_lr_head
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting Phase A training...\n")
    trainer.train()
    
    # Save
    phase_a_path = os.path.join(config.output_dir, "phase_a_checkpoint")
    model.save_pretrained(phase_a_path)
    processor.save_pretrained(phase_a_path)
    print(f"\n✓ Phase A model saved to {phase_a_path}\n")
    
    return model


def phase_b_training(
    model: MMSForMultilingualSER,
    english_train_df: pd.DataFrame,
    english_val_df: pd.DataFrame,
    tamil_train_df: pd.DataFrame,
    tamil_val_df: pd.DataFrame,
    sinhala_train_df: pd.DataFrame,
    sinhala_val_df: pd.DataFrame,
    processor: Wav2Vec2FeatureExtractor,
    config: Config
):
    """
    Phase B: Joint multilingual training
    Goal: Adapt to all languages with higher sampling for Tamil/Sinhala
    """
    print("\n" + "="*80)
    print("PHASE B: MULTILINGUAL JOINT TRAINING")
    print("="*80 + "\n")
    
    # Create datasets for all languages
    train_datasets = [
        MultilingualEmotionDataset(english_train_df, processor, config, 'english', augment=True),
        MultilingualEmotionDataset(tamil_train_df, processor, config, 'tamil', augment=True),
        MultilingualEmotionDataset(sinhala_train_df, processor, config, 'sinhala', augment=True)
    ]
    
    val_datasets = [
        MultilingualEmotionDataset(english_val_df, processor, config, 'english', augment=False),
        MultilingualEmotionDataset(tamil_val_df, processor, config, 'tamil', augment=False),
        MultilingualEmotionDataset(sinhala_val_df, processor, config, 'sinhala', augment=False)
    ]
    
    print("Training samples:")
    print(f"  English: {len(train_datasets[0])}")
    print(f"  Tamil: {len(train_datasets[1])}")
    print(f"  Sinhala: {len(train_datasets[2])}")
    print(f"\nSampling weights:")
    print(f"  English: {config.lang_weights['english']}x")
    print(f"  Tamil: {config.lang_weights['tamil']}x")
    print(f"  Sinhala: {config.lang_weights['sinhala']}x\n")
    
    # Combine datasets
    train_dataset = CombinedDataset(train_datasets)
    val_dataset = CombinedDataset(val_datasets)
    
    # Weighted sampler for balanced language sampling
    sampler = create_weighted_sampler(train_datasets, config)
    data_collator = DataCollatorForMultilingualSER(processor=processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "phase_b"),
        per_device_train_batch_size=config.phase_b_batch_size,
        per_device_eval_batch_size=config.phase_b_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.phase_b_epochs,
        learning_rate=config.phase_b_lr_head,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False
    )
    
    # Differential learning rates (lower than Phase A)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'wav2vec2' in n and p.requires_grad],
            'lr': config.phase_b_lr_backbone
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(x in n for x in ['lid_layer', 'emotion_head'])],
            'lr': config.phase_b_lr_head
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    # Override sampler
    trainer._get_train_sampler = lambda: sampler
    
    # Train
    print("Starting Phase B training...\n")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Final model saved to {final_path}\n")
    
    return model


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("MULTILINGUAL SER WITH MMS-1B + LANGUAGE IDENTIFICATION")
    print("="*80 + "\n")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: GPU not detected!")
        print("MMS-1B requires GPU for practical training.")
        print("This will be VERY SLOW on CPU.\n")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Load processor
    print("Loading MMS-1B processor...")
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
        print("✓ Processor loaded\n")
    except Exception as e:
        print(f"Error loading processor: {e}")
        print("Make sure you have internet connection and sufficient disk space.")
        return
    
    # Load data (placeholder - adapt to your data structure)
    print("Loading datasets...")
    print("⚠️  NOTE: Update the data loading section with your actual dataset paths\n")
    
    # Example: Load your prepared datasets
    # For demonstration, create dummy dataframes
    # REPLACE THIS with your actual data loading code
    
    # Option 1: Load from CSV
    # english_df = pd.read_csv("data/english_emotions.csv")  # columns: path, label
    # tamil_df = pd.read_csv("data/tamil_emotions.csv")
    # sinhala_df = pd.read_csv("data/sinhala_emotions.csv")
    
    # Option 2: Load from folders (example)
    # See prepare_datasets.py or kaggle_multilingual_ser.py for examples
    
    # For now, exit with instructions
    print("="*80)
    print("DATA LOADING REQUIRED")
    print("="*80)
    print("\nBefore running training, you need to:")
    print("1. Prepare your datasets (English, Tamil, Sinhala)")
    print("2. Create CSV files or dataframes with columns: ['path', 'label']")
    print("3. Update the data loading section in this script")
    print("\nLabel mapping: 0=neutral, 1=happy, 2=sad, 3=angry, 4=fear")
    print("\nExample:")
    print("  english_df = pd.read_csv('data/english_emotions.csv')")
    print("  tamil_df = pd.read_csv('data/tamil_emotions.csv')")
    print("  sinhala_df = pd.read_csv('data/sinhala_emotions.csv')")
    print("\n" + "="*80)
    
    # Uncomment and modify when data is ready:
    """
    # Split data
    english_train, english_val = train_test_split(
        english_df, test_size=0.15, random_state=42, stratify=english_df['label']
    )
    tamil_train, tamil_val = train_test_split(
        tamil_df, test_size=0.15, random_state=42, stratify=tamil_df['label']
    )
    sinhala_train, sinhala_val = train_test_split(
        sinhala_df, test_size=0.15, random_state=42, stratify=sinhala_df['label']
    )
    
    # Initialize model
    print("Initializing MMS-1B model...")
    model = MMSForMultilingualSER(config)
    print("✓ Model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")
    
    # Phase A: English-only training
    model = phase_a_training(
        model, english_train, english_val, processor, config
    )
    
    # Phase B: Multilingual training
    model = phase_b_training(
        model,
        english_train, english_val,
        tamil_train, tamil_val,
        sinhala_train, sinhala_val,
        processor, config
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE! 🎉")
    print("="*80)
    print(f"\nFinal model saved at: {os.path.join(config.output_dir, 'final_model')}")
    print("\nNext steps:")
    print("1. Evaluate on test sets")
    print("2. Test language identification accuracy")
    print("3. Analyze per-language emotion recognition performance")
    """


if __name__ == "__main__":
    main()
