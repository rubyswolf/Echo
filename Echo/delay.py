import soundfile as sf
import numpy as np

# Load input audio
input_file = "input2.wav"
output_file = "output2.wav"
audio, sr = sf.read(input_file)

# Ensure mono
if audio.ndim > 1:
    audio = audio[:, 0]


# 5ms delay
delay_samples = 50
# Create delayed signal
delayed_signal = np.zeros_like(audio)
delayed_signal[delay_samples:] = audio[:-delay_samples]
# Write output
sf.write(output_file, delayed_signal, sr)
print(f"Created {output_file}")
