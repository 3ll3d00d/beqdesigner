import numpy as np
from matplotlib import pyplot as plt

from model.jriver.filter import MDSFilter

opt_55 = MDSFilter(4, 55)

f = plt.figure()
# plt.ylim(-50, 0)
plt.grid(visible=True)
plt.xlim(2.0, 2.1)

t, sr_l = opt_55.lp_output.step_response()
_, sr_h = opt_55.hp_output.step_response()
norm_to = np.max(sr_l)
sr_h = sr_h * (100.0 / norm_to)
sr_l = sr_l * (100.0 / norm_to)

plt.plot(t, sr_l, label='lp')
plt.plot(t, sr_h, label='hp')
plt.legend()
plt.show()

