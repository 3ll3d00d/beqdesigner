from matplotlib import pyplot as plt

from model.iir import FilterType, ComplexLowPass, ComplexHighPass
from model.jriver import JRIVER_FS
from model.jriver.filter import MultiwayCrossover, MDSXO, StandardXO

multimds = MultiwayCrossover('L', [
    MDSXO(4, 100, lp_channel='L', hp_channel='R'),
    MDSXO(4, 350, lp_channel='R', hp_channel='C'),
    MDSXO(4, 1200, lp_channel='C', hp_channel='SW'),
], [None] * 4, fs=48000)
multistd = MultiwayCrossover('L', [
    StandardXO('L', 'R',
               ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 100),
               ComplexHighPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 100)),
    StandardXO('R', 'C',
               ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 350),
               ComplexHighPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 350)),
    StandardXO('C', 'SW',
               ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 1250),
               ComplexHighPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 1250)),
], [None] * 4)
multimix = MultiwayCrossover('L', [
    MDSXO(4, 100, lp_channel='L', hp_channel='R'),
    StandardXO('R', 'C',
               ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 1250),
               ComplexHighPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 1250)),
    StandardXO('C', 'SW',
               ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 5000),
               ComplexHighPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 5000)),
], [None] * 4)

multi = multimds
f = plt.figure()
plt.grid(visible=True)
# plt.ylim(-85, -10)
plt.xlim(20, 20000)
plt.xscale('log')
# plt.xlim(2.055, 2.065)
# plt.xlim(2.01, 2.03)

vals = multi.output
for i, v in enumerate(vals):
    plt.plot(*v.avg, label=f"W{i} - {v.name}")
    # plt.plot(*v.step_response(), label=f"W{i} - {v.name}")
sum = multi.sum
# plt.plot(*sum.avg, label='sum')
# plt.plot(*sum.step_response(), label='sum')

print('\n'.join((str(f) for f in multi.graph.filters)))

print(multi.graph.render(vertical=False))

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
