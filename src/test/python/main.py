import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fs = 48000



# Frequency response
w, h = signal.freqz(b=bq.b, a=bq.a, worN=65536)
# Generate frequency axis
w = w * fs / (2 * np.pi)
# Plot
plt.semilogx(w, 20 * np.log10(np.abs(h)), 'b')
plt.ylabel('Amplitude', color='b')
plt.xlabel('Frequency')
plt.show()
