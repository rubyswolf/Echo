import pedalboard
from pedalboard import Reverb
import soundfile as sf
import numpy as np

# ===== Load audio =====
input_file = 'input2.wav'
output_file = 'output2.wav'
audio, sr = sf.read(input_file)

# Ensure mono and float32
if audio.ndim > 1:
    audio = audio[:, 0]
audio = audio.astype(np.float32)

# ===== Create pedalboard with reverb =====
# Parameters for a richer reverb
reverb = Reverb(
    room_size=0.5,   # Larger room
    damping=0.5,     # Less damping for brighter reflections
    wet_level=1.0,   # More wet signal
    dry_level=0.0,   # Less dry signal
    width=1.0,
    freeze_mode=0
)

board = pedalboard.Pedalboard([reverb])

# ===== Apply reverb =====
audio_with_reverb = board(audio, sample_rate=sr)

# ===== Save output =====
sf.write(output_file, audio_with_reverb, sr)
print(f"Reverb applied with wet_level=0.7 and saved to {output_file}")
