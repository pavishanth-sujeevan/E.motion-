import os
import shutil
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# Fix for legacy checkpoint loading
from numpy.core.multiarray import _reconstruct

torch.serialization.add_safe_globals([_reconstruct])


def main():
    # --- 1. CONFIGURATION PATHS ---
    PRETRAINED_CHECKPOINT = r"C:\TTS-VITS-tam\best_model.pth"
    DATA_PATH = r"C:\TTS-VITS-tam\tamil_vits"
    OUTPUT_PATH = r"C:\TTS-VITS-tam\finetuned_output"

    # --- 2. DATASET SETTINGS ---
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="tamil_vits",
        path=DATA_PATH,
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_val.csv",
        language="ta",
    )

    # --- 3. MODEL & TRAINING SETTINGS ---
    config = VitsConfig(
        batch_size=4,
        eval_batch_size=2,
        num_loader_workers=0,  # Set to 0 to prevent WinError 32 Permission errors
        num_eval_loader_workers=0,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=100,
        lr=0.0001,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        characters=CharactersConfig(
            characters="ஂஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்",
            punctuations=r".,!?;:-\"()",
            pad="_",
            eos="~",
            bos="^",
            blank=" ",
        ),
        datasets=[dataset_config],
        output_path=OUTPUT_PATH,
    )

    # --- 4. INITIALIZE COMPONENTS ---
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load data samples
    train_samples, eval_samples = load_tts_samples(config.datasets[0], eval_split=True)

    # Initialize model with the NEW Tamil vocabulary size
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # --- 5. MANUAL WEIGHT LOADING (The "Hack") ---
    if os.path.exists(PRETRAINED_CHECKPOINT):
        print(f" > Loading pretrained weights from {PRETRAINED_CHECKPOINT}...")
        # Load the file manually to strip the incompatible embedding layer
        checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # We delete the embedding layer because its size (English/Multi)
        # doesn't match our new Tamil character list.
        if "text_encoder.emb.weight" in state_dict:
            print(" > Stripping old embedding layer to accommodate Tamil script.")
            del state_dict["text_encoder.emb.weight"]

        # Load everything else (the voice, the rhythms, the flow)
        model.load_state_dict(state_dict, strict=False)

    # --- 6. START TRAINER ---
    # We set restore_path=None so the Trainer doesn't try to reload and crash
    last_run_folder= r"C:\TTS-VITS-tam\finetuned_output\run-February-20-2026_10+27AM-0000000"

    trainer = Trainer(
        TrainerArgs(
            continue_path=last_run_folder
        ),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )

    trainer.fit()


if __name__ == "__main__":
    main()