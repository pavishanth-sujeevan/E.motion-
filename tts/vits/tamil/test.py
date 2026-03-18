import os
from TTS.utils.synthesizer import Synthesizer

# 1. Update these paths to your actual folder!
MODEL_PATH = r"C:\TTS-VITS-tam\finetuned_output\run-February-20-2026_10+27AM-0000000\best_model.pth"
CONFIG_PATH = r"C:\TTS-VITS-tam\finetuned_output\run-February-20-2026_10+27AM-0000000\config.json"
OUT_PATH = "test_output.wav"

# 2. Initialize the Synthesizer
# If you don't have a GPU available for testing, set use_cuda=False
syn = Synthesizer(
    tts_checkpoint=MODEL_PATH,
    tts_config_path=CONFIG_PATH,
    use_cuda=True
)

# 3. Generate the speech
text = "முதலில் சேரும் ஆயிரம் பேருக்கு பயிற்சி கட்டணத்தில் ஐம்பது விழுக்காடு சலுகை வழங்கப்படும்"
print("Synthesizing...")
outputs = syn.tts(text)

# 4. Save to file
syn.save_wav(outputs, OUT_PATH)
print(f"Done! Check {OUT_PATH}")