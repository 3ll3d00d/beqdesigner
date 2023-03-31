import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from model.iir import ComplexLowPass, FilterType
from model.jriver.filter import MDSXO, FilterGraph, convert_filter_to_mc_dsp, Delay

o = MDSXO(4, 100, lp_channel=['L'], hp_channel=['R'])
o.in_channel = 'L'
o.recalc(force=True)

y = ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, 48000, 120, q=0.7071)
d = Delay({**Delay.default_values(),
           'Channels': '2',
           'Delay': '0.0'})
g = FilterGraph(0, ['L'], ['L'], [convert_filter_to_mc_dsp(y, '2'), d], regen=True)
yo = g.simulate(analysis_resolution=0.1)

print(f"DC GD : mds {o.delay:.3f} lpf: {y.dc_gd_millis:.3f} delta: {o.delay - y.dc_gd_millis:.3f}")

f = plt.figure()
# plt.ylim(-40, 10)
plt.grid(visible=True)
plt.xlim(10, 400)
# plt.ylim(-10, 10)
# plt.xscale('log')
# plt.plot(o.lp_output.avg[0], o.lp_output.avg[1], label=f"lp")
# plt.plot(o.hp_output.avg[0], o.hp_output.avg[1], label=f"lp")

# t, mds_sr = o.lp_output
# _, lr_sr = yo['L'].step_response()
t, mds_sr = o.lp_output.waveform
_, lr_sr = yo['L'].waveform

mds_peak = o.lp_output.peak_pos
lpf_peak = yo['L'].peak_pos
delta = (mds_peak - lpf_peak) * 1000
print(f"Peak : mds: {mds_peak:.3f} lpf: {lpf_peak:.3f} delta: {delta:.3f}")

shifted_lpf = yo['L'].shift(delta)
_, lr_sr = shifted_lpf.waveform

w, h = signal.freqz(shifted_lpf.raw())

x = w * 48000 * 1.0 / (2 * np.pi)
y = 20 * np.log10(abs(h))
y1 = np.angle(h, deg=True)
# plt.semilogx(x, y, label='lr')
plt.semilogx(x, y1, label='lrp')

w, h = signal.freqz(o.lp_output.raw())
x = w * 48000 * 1.0 / (2 * np.pi)
y = 20 * np.log10(abs(h))
y1 = np.angle(h, deg=True)
# plt.semilogx(x, y, label='mds')
plt.semilogx(x, y1, label='mdsp')

# t, sr_l = opt_55.lp_output.step_response()
# _, sr_h = opt_55.hp_output.step_response()
# mds_sr = mds_sr * (100.0 / np.max(mds_sr))
# lr_sr = lr_sr * (100.0 / np.max(lr_sr))

# plt.plot(t, mds_sr, label='mds')
# plt.plot(t, lr_sr, label='lr')
plt.legend()
plt.show()
