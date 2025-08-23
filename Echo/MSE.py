import soundfile as sf
import numpy as np

# Load audio files
orig, sr1 = sf.read("output2.wav")
out, sr2 = sf.read("echo_output2.wav")

# Ensure sample rates match
if sr1 != sr2:
    raise ValueError(f"Sample rates differ: {sr1} vs {sr2}")

# Match lengths
min_len = min(len(orig), len(out))
orig = orig[:min_len]
out = out[:min_len]

# Compute Mean Squared Error
mse = np.mean((orig - out) ** 2)
print(f"MSE = {mse:.10f}")