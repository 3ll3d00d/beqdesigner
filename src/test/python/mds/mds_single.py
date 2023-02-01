import numpy as np
from matplotlib import pyplot as plt

from model.jriver.filter import MDSXO

o = MDSXO(4, 100)

f = plt.figure()
plt.ylim(-70, -10)
plt.grid(visible=True)
plt.xlim(10, 1280)
plt.xscale('log')
plt.plot(o.lp_output.avg[0], o.lp_output.avg[1], label=f"lp")
plt.plot(o.hp_output.avg[0], o.hp_output.avg[1], label=f"lp")
plt.legend()

# t, sr_l = opt_55.lp_output.step_response()
# _, sr_h = opt_55.hp_output.step_response()
# norm_to = np.max(sr_l)
# sr_h = sr_h * (100.0 / norm_to)
# sr_l = sr_l * (100.0 / norm_to)
#
# plt.plot(t, sr_l, label='lp')
# plt.plot(t, sr_h, label='hp')
plt.show()

