import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.configs.shared_configs import BaseDatasetConfig

# --- PATHS & PRE-TRAINED MODEL ---
# Point this to the folder containing your pre-trained 'best_model.pth' and 'config.json'
PRETRAINED_PATH = r"C:\TTS-Project\Base_Model"
RESTORE_PATH = os.path.join(PRETRAINED_PATH, "best_model.pth")

DATASET_PATH = r"C:\TTS-Project\VITS_dataset"  # Your training data
OUTPUT_PATH = r"C:\TTS-Project\checkpoints\vits_finetune_output"


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1. DATASET CONFIG
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        path=DATASET_PATH
    )

    # 2. AUDIO CONFIG (Must match your training data exactly!)
    audio_config = VitsAudioConfig(
        sample_rate=22050,  # Ensure your WAVs are actually 22050
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    # 3. VITS MODEL CONFIG
    config = VitsConfig(
        audio=audio_config,
        run_name="vits_finetune_run",
        batch_size=4,  # Adjust based on your GPU VRAM
        eval_batch_size=2,
        batch_group_size=0,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        max_audio_len=150000,
        test_delay_epochs=-1,
        epochs=1000,
        save_step=500,  # SAVE EVERY 500 STEPS (Critical!)
        save_n_checkpoints=5,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
        datasets=[dataset_config],
        output_path=OUTPUT_PATH,
        save_best_after=10,
        # Fine-tuning specific: keep learning rate lower than scratch training
        lr=0.0001,
        test_sentences=[
            "The quick brown fox jumps over the lazy dog.",
            "I am currently fine tuning this voice model and it is starting to sound better.",
            "Synthesis quality depends on the mel loss and the duration predictor."
        ],
    )

    # 4. INITIALIZE COMPONENTS
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # 5. LOAD DATA
    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.01)

    # 6. INITIALIZE MODEL & RESTORE WEIGHTS
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # 7. TRAINER
    trainer = Trainer(
        TrainerArgs(),  # THIS IS THE KEY FOR FINETUNING
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()