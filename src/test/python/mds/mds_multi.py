from matplotlib import pyplot as plt

from model.jriver.filter import MDSFilter

multi = MDSFilter([(4, 100, 'L'), (4, 350, 'R')], (4, 1250, 'C', 'SL'))
# multi = MDSFilter([], (4, 153, 'L', 'R'))

f = plt.figure()
# plt.ylim(-85, -10)
plt.grid(visible=True)
# plt.xlim(20, 20000)
plt.xlim(2.00, 2.04)
# plt.xscale('log')

vals = []
for i in range(0, 4):
    c, o = multi.output(i)
    # plt.plot(o.avg[0], o.avg[1], label=f"W{i} - {c}")
    plt.plot(*o.step_response(), label=f"W{i} - {c}")

print('\n'.join((str(f) for f in multi.graph.filters)))

# t, sr_l = multi.lp_output.step_response()
# _, sr_h = multi.hp_output.step_response()
# norm_to = np.max(sr_l)
# sr_h = sr_h * (100.0 / norm_to)
# sr_l = sr_l * (100.0 / norm_to)
#
# plt.plot(t, sr_l, label='lp')
# plt.plot(t, sr_h, label='hp')
plt.legend()
plt.show()

