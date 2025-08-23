import soundfile as sf
from scipy.signal import butter, lfilter

# Load input audio
input_file = "input2.wav"
output_file = "output2.wav"
audio, sr = sf.read(input_file)

# Ensure mono
if audio.ndim > 1:
    audio = audio[:, 0]

# ===== Design a lowpass filter =====
# Butterworth lowpass
cutoff_hz = 4000  # Adjust cutoff frequency (Hz)
order = 4

nyq = 0.5 * sr
normal_cutoff = cutoff_hz / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Apply filter
filtered = lfilter(b, a, audio)

# Write output
sf.write(output_file, filtered, sr)
print(f"Created {output_file} (lowpass @ {cutoff_hz} Hz)")
