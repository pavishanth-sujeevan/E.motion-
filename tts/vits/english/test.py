import os
from TTS.api import TTS

# 1. Update this to your EXACT folder name
model_folder = r"C:\TTS-Project\checkpoints\vits_finetune_output\vits_finetune_run-March-13-2026_10+39AM-0000000"

# 2. Point to the specific files
model_path = os.path.join(model_folder, "best_model.pth")
config_path = os.path.join(model_folder, "config.json")

print("--- Loading Model (This may take a minute) ---")
# If you don't have a GPU, remove .to("cuda")
tts = TTS(model_path=model_path, config_path=config_path).to("cuda")

print("--- Generating Audio ---")
output_file = "final_accuracy_test.wav"
tts.tts_to_file(text="if you feel that you'd like to tell me what has hurt you.i'll be glad to listen.", file_path=output_file, lengnth_scale=2.0)

print(f"--- SUCCESS! Listen to: {os.path.abspath(output_file)} ---")
