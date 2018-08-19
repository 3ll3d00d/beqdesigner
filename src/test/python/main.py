import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fs = 48000

f = 10
b, a = signal.butter(1, f/(0.5*fs), btype='high')
sos = signal.butter(12, f/(0.5*fs), btype='high', output='sos')
print(f"b {b} a {a}")
print(f"sos {sos}")

# Frequency response
w, h = signal.freqz(b=b, a=a, worN=65536)
# Generate frequency axis
w = w * fs / (2 * np.pi)
# Plot
plt.semilogx(w, 20 * np.log10(np.abs(h)), 'b')
plt.ylabel('Amplitude', color='b')
plt.xlabel('Frequency')
plt.show()
