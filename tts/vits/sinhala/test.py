import sys
import types
import torch
import transformers

# 1. Force-create the missing import_utils sub-module
if not hasattr(transformers.utils, "import_utils"):
    transformers.utils.import_utils = types.ModuleType("import_utils")
    sys.modules["transformers.utils.import_utils"] = transformers.utils.import_utils

# 2. Inject all the functions TTS expects to find there
import_utils = transformers.utils.import_utils
funcs_to_patch = [
    'is_torch_greater_or_equal',
    'is_torch_available',
    'is_torchcodec_available',
    'is_inflection_available'
]

for func in funcs_to_patch:
    if not hasattr(import_utils, func):
        # We set these to return False or True depending on typical needs
        setattr(import_utils, func, lambda *args, **kwargs: False)

# 3. Patch the second location (pytorch_utils)
if not hasattr(transformers, "pytorch_utils"):
    transformers.pytorch_utils = types.ModuleType("pytorch_utils")
    sys.modules["transformers.pytorch_utils"] = transformers.pytorch_utils

if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
    setattr(transformers.pytorch_utils, "isin_mps_friendly", torch.isin)

# NOW you can safely import TTS
from TTS.tts.configs.vits_config import VitsConfig
import sys
import types
import torch
import transformers.utils

# 1. Patch import_utils
if not hasattr(transformers.utils, "import_utils"):
    sys.modules["transformers.utils.import_utils"] = types.ModuleType("import_utils")
import transformers.utils.import_utils as import_utils

for func in ['is_torch_greater_or_equal', 'is_torch_available', 'is_inflection_available']:
    if not hasattr(import_utils, func):
        setattr(import_utils, func, lambda *args, **kwargs: True)

# 2. Patch pytorch_utils (The isin_mps_friendly fix)
import transformers.pytorch_utils as pytorch_utils
if not hasattr(pytorch_utils, "isin_mps_friendly"):
    setattr(pytorch_utils, "isin_mps_friendly", torch.isin)

# NOW you can do your regular imports
from TTS.tts.configs.vits_config import VitsConfig
# ... rest of your code
# Create the missing sub-module if it doesn't exist
if not hasattr(transformers.utils, "import_utils"):
    sys.modules["transformers.utils.import_utils"] = types.ModuleType("import_utils")

import transformers.utils.import_utils as import_utils

# Define the missing functions that TTS is screaming for
missing_funcs = ['is_torch_greater_or_equal', 'is_torchcodec_available', 'is_torch_available', 'is_inflection_available']
for func in missing_funcs:
    if not hasattr(import_utils, func):
        setattr(import_utils, func, lambda *args, **kwargs: True) # Force return True for availability checks

from TTS.tts.configs.vits_config import VitsConfig
import transformers.pytorch_utils

# Patch for the 'isin_mps_friendly' error
if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
    # We define it manually so TTS can find it
    def isin_mps_friendly(elements, test_elements):
        import torch
        return torch.isin(elements, test_elements)


    transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly
import os
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# 1. Update these paths to your best checkpoint
MODEL_PATH = r"C:\TTS-VITS-sin\output\sinhala_finetuned\female_sinhala_finetune-February-23-2026_10+22AM-0000000\best_model.pth"
CONFIG_PATH = r"C:\TTS-VITS-sin\output\sinhala_finetuned\female_sinhala_finetune-February-23-2026_10+22AM-0000000\config.json"
OUT_FILE = "sample-ai.wav"
TEXT = "ආපත්ති විනිශ්චය දුෂ්කර කරුණෙකි ප්රාතිමෝක්ෂය පාඩම් කර ඇති පමණින් හෝ එයට අර්ථ අස්වා ඇති පමණින් හෝ ආපත්ති විනිශ්චයට නො යා යුතු ය" # "Hello, I speak Sinhala."

# 2. Load Config and Model
config = VitsConfig()
config.load_json(CONFIG_PATH)

ap = AudioProcessor.init_from_config(config)
tokenizer, encoder = TTSTokenizer.init_from_config(config)

model = Vits(config, ap, tokenizer, speaker_manager=None)
model.load_checkpoint(config, MODEL_PATH, eval=True)

# 3. Inference
# Use torch.no_grad() to save memory and speed up
# NEW CODE
# ... [Paste your transliteration code/classes here] ...
from romanizer import sinhala_to_roman

# Now this will work:


TEXT_SINHALA = "ආපත්ති විනිශ්චය දුෂ්කර කරුණෙකි ප්රාතිමෝක්ෂය පාඩම් කර ඇති පමණින් හෝ එයට අර්ථ අස්වා ඇති පමණින් හෝ ආපත්ති විනිශ්චයට නො යා යුතු ය."

# 1. Convert Sinhala to Romanized text using your function
TEXT_ROMAN = sinhala_to_roman(TEXT_SINHALA)
print(f"Romanized Text: {TEXT_ROMAN}")

# 2. Tokenize the ROMANIZED text
# Now the tokenizer will find 'a', 'y', 'u', etc. in the vocabulary
tokens = tokenizer.text_to_ids(TEXT_ROMAN)

# 3. Create tensor and run inference
text_tensor = torch.LongTensor(tokens).unsqueeze(0).to(model.device)

with torch.no_grad():
    outputs = model.inference(text_tensor, config)

    # Handle the 'wav' vs dictionary issue safely
    if isinstance(outputs, dict):
        audio = outputs.get('wav') or outputs.get('model_outputs')
    else:
        audio = outputs

# 4. Save
if audio is not None:
    ap.save_wav(audio.cpu().numpy().flatten(), "ai_voice.wav")
    print("Audio saved successfully!")