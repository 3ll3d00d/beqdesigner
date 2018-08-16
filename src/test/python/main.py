import numpy as np
from matplotlib import pyplot as mp
from scipy import signal
from yodel.filter import Biquad

input = signal.unit_impulse(48000, idx='mid')
output = np.zeros(48000)
bq = Biquad()
bq.low_shelf(48000, 20, 0.707, 10)
bq.process(input, output)

nperseg = min(1 << (48000 - 1).bit_length(), output.shape[-1])
f, Pxx_spec = signal.welch(output, 48000, nperseg=nperseg, scaling='spectrum', detrend=False)
Pxx_spec = np.sqrt(Pxx_spec)

mp.figure()
mp.plot(f, 20.0 * np.log10(Pxx_spec))
mp.xscale('log')
mp.xlim(5, 200)
mp.show()
