import sys
import transformers.utils

from romanizer import sinhala_to_roman

# 1. Ensure the import_utils sub-module is recognized
if not hasattr(transformers.utils, "import_utils"):
    import types

    sys.modules["transformers.utils.import_utils"] = types.ModuleType("import_utils")
    import transformers.utils.import_utils as import_utils
else:
    from transformers.utils import import_utils

# 2. Map functions with 'dummy' versions that accept arguments (*args)
# This prevents the "takes 0 positional arguments but 1 was given" error
missing_functions = [
    'is_torch_greater_or_equal',
    'is_torchcodec_available',
    'is_torch_available',
    'is_inflection_available'
]

for func_name in missing_functions:
    if not hasattr(import_utils, func_name):
        # We look for the function in main utils; if not found,
        # we provide a lambda that accepts any number of arguments.
        actual_func = getattr(transformers.utils, func_name, lambda *args, **kwargs: False)
        setattr(import_utils, func_name, actual_func)
import os
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.datasets import load_tts_samples
from types import SimpleNamespace
from TTS.tts.utils.text.tokenizer import TTSTokenizer


if __name__ == '__main__':
    # 1. Paths
    DATASET_PATH = r"C:\TTS-VITS-sin\sinhala_vits" # wavs/ and metadata.csv
    BASE_MODEL_PATH = "base_model/Sanuki_190000.pth"
    BASE_CONFIG_PATH = "base_model/Sanuki_config.json"
    OUT_PATH = "output/sinhala_finetuned"

    # 2. Load and Configure
    config = VitsConfig()
    config.load_json(BASE_CONFIG_PATH)

    # Critical: Update config for fine-tuning
    config.run_name = "female_sinhala_finetune"
    config.output_path = OUT_PATH
    config.batch_size = 12         # Adjust based on your GPU VRAM
    config.eval_batch_size = 4
    config.epochs = 300            # Fewer epochs needed for fine-tuning
    config.lr = 0.00005            # Lower learning rate to avoid over-fitting

    class Map(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    # Link your new dataset (ensure metadata.csv is pipe-separated)
    dataset_info = {
        "name": "ljspeech",
        "dataset_name": "ljspeech",
        "path": DATASET_PATH,
        "meta_file_train": "metadata_train.csv",
        "meta_file_val": "",      # <--- Add this (stops the KeyError)
        "formatter": "ljspeech",
        "language": "si",
        "ignored_speakers": None,
        "meta_file_attn_mask": ""# Good practice to define the language (Sinhala)
    }

    config.datasets = [Map(dataset_info)]

    # 3. Init Model and Audio
    tokenizer, encoder = TTSTokenizer.init_from_config(config)
    ap = AudioProcessor.init_from_config(config)
    model = Vits(config, ap, tokenizer=tokenizer, speaker_manager=None)

    # 4. Load Dataset Samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_batch_size
    )

    # 5. Start Training
    trainer = Trainer(
        TrainerArgs(continue_path=r"C:\TTS-VITS-sin\output\sinhala_finetuned\female_sinhala_finetune-February-23-2026_10+22AM-0000000"), # Load the Sanuki female weights
        config,
        OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )


    # --- THE CONNECTION ---
    def apply_romanization(samples):
        for sample in samples:
            # 'text' is the key used by TTS datasets
            sample['text'] = sinhala_to_roman(sample['text'])
        return samples


    train_samples = apply_romanization(train_samples)
    eval_samples = apply_romanization(eval_samples)

    config.save_step = 100  # Save a checkpoint every 500 steps
    config.save_checkpoints = True
    config.print_step = 25
    # Show progress in console every 10 steps

    trainer.fit()