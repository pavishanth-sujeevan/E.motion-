# CONFIG FOR MMS-300M (RECOMMENDED FOR KAGGLE T4)
# Copy this cell and replace your current Config cell

@dataclass
class Config:
    # Model - USING MMS-300M (smaller, faster, reliable)
    model_name: str = "facebook/mms-300m"  # Changed from mms-1b-all
    num_labels: int = 5
    num_languages: int = 3
    hidden_size: int = 1024  # MMS-300M hidden size (vs 1280 for 1B)
    dropout: float = 0.1
    freeze_layers: int = 15  # MMS-300M has ~24 layers (vs 48 for 1B)
    
    # Language mapping
    lang_to_id: Dict[str, int] = None
    id_to_lang: Dict[int, str] = None
    
    # Audio
    sampling_rate: int = 16000
    max_duration: float = 10.0  # Can use full 10s with 300M
    
    # Phase A (English only)
    phase_a_epochs: int = 10
    phase_a_batch_size: int = 8  # Can use larger batch!
    phase_a_lr_backbone: float = 5e-6
    phase_a_lr_head: float = 1e-4
    
    # Phase B (Multilingual)
    phase_b_epochs: int = 15
    phase_b_batch_size: int = 8  # Can use larger batch!
    phase_b_lr_backbone: float = 3e-6
    phase_b_lr_head: float = 5e-5
    
    # Training
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4  # Back to original
    
    # Language weights
    lang_weights: Dict[str, float] = None
    
    # Output
    output_dir: str = "/kaggle/working/mms_multilingual_ser"
    
    def __post_init__(self):
        if self.lang_to_id is None:
            self.lang_to_id = {'english': 0, 'tamil': 1, 'sinhala': 2}
            self.id_to_lang = {v: k for k, v in self.lang_to_id.items()}
        
        if self.lang_weights is None:
            self.lang_weights = {
                'english': 1.0,
                'tamil': 3.0,
                'sinhala': 5.0
            }

config = Config()
os.makedirs(config.output_dir, exist_ok=True)

print("="*80)
print("✓ USING MMS-300M (MEMORY-EFFICIENT CONFIGURATION)")
print("="*80)
print(f"Model: {config.model_name}")
print(f"Parameters: ~300M (vs 1B)")
print(f"Hidden size: {config.hidden_size}")
print(f"Batch size: {config.phase_a_batch_size}")
print(f"Effective batch: {config.phase_a_batch_size * config.gradient_accumulation_steps}")
print(f"Expected memory: ~5-6 GB (plenty of room!)")
print(f"Expected training time: 4-5 hours")
print("="*80)
