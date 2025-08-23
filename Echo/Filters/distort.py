import soundfile as sf
import numpy as np

# Load input audio
input_file = "input2.wav"
output_file = "output2.wav"
audio, sr = sf.read(input_file)

# Ensure mono
if audio.ndim > 1:
    audio = audio[:, 0]

gain = 7.0

# Waveshaping filter
filtered = np.tanh(audio*gain)/gain
# Write output
sf.write(output_file, filtered, sr)
print(f"Created {output_file}")
