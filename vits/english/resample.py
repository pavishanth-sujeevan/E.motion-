import os
from pydub import AudioSegment
from pathlib import Path


def process_dataset(input_dir, output_dir, target_hz=22050):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Supported extensions
    extensions = ('.wav', '.mp3', '.flac', '.ogg')

    for file_path in Path(input_dir).rglob('*'):
        if file_path.suffix.lower() in extensions:
            # Create the same sub-directory structure in the output folder
            relative_path = file_path.relative_to(input_dir)
            new_export_path = Path(output_dir) / relative_path
            new_export_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Load and Resample
                audio = AudioSegment.from_file(str(file_path))
                resampled = audio.set_frame_rate(target_hz)

                # Export (Converting all to .wav for consistency in datasets)
                resampled.export(new_export_path.with_suffix('.wav'), format="wav")
                print(f"Processed: {file_path.name}")
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")


# Configuration
SOURCE_FOLDER = "C:\TTS-Project\XTTS_Dataset\wavs"
DESTINATION_FOLDER = "C:\TTS-Project\XTTS_Dataset\wa"

process_dataset(SOURCE_FOLDER, DESTINATION_FOLDER)