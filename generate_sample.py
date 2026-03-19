"""
Generates a sample WAV audio file (440 Hz tone + harmonics) for testing the AI Sound Analyzer.
Run: python generate_sample.py
"""
import numpy as np
import soundfile as sf

sr = 22050
duration = 3.0  # seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# A4 note = 440 Hz + harmonics (makes it richer for spectrogram)
audio = (
    0.5  * np.sin(2 * np.pi * 440 * t) +   # fundamental
    0.25 * np.sin(2 * np.pi * 880 * t) +   # 2nd harmonic
    0.15 * np.sin(2 * np.pi * 1320 * t) +  # 3rd harmonic
    0.08 * np.sin(2 * np.pi * 1760 * t)    # 4th harmonic
)

# Normalise to [-1, 1]
audio = audio / np.max(np.abs(audio))

sf.write("sample_tone.wav", audio, sr)
print("✅  sample_tone.wav created successfully!")
