import math

import numpy as np

from model.iir import MDS_FREQ_DIVISOR
from model.jriver.filter import MDSFilter
import matplotlib.pyplot as plt

f = plt.figure()
plt.ylim(-50, 10)
plt.grid(visible=True)


def solve(order: int, fc: float) -> MDSFilter:
    attempts = []
    factor = MDS_FREQ_DIVISOR[order]
    while True:
        mds = MDSFilter(order, fc, factor)
        attempts.append(mds)
        if mds.actual_fc == 0.0:
            print(f"NO CROSSING FOUND {fc} - {mds}")
            plt.plot(mds.lp_output.avg[0], mds.lp_output.avg[1], label=f'lp {fc}')
            plt.plot(mds.hp_output.avg[0], mds.hp_output.avg[1], label=f'hp {fc}')
            plt.show()
            break
        else:
            factor = mds.optimised_divisor
            if len(attempts) > 6:
                delta_fc_1 = abs(attempts[-2].fc_delta)
                delta_fc_2 = abs(attempts[-1].fc_delta)
                print(f"Unable to reach target after {len(attempts)} for {order}/{fc:.2f}, discarding last [{delta_fc_1:.2f} vs {delta_fc_2:.2f}]")
                break
            elif len(attempts) > 1:
                delta_fc_1 = attempts[-2].fc_delta
                delta_fc_2 = attempts[-1].fc_delta
                if math.isclose(round(delta_fc_1, 1), round(delta_fc_2, 1)):
                    attempts.pop()
                    print(f"Optimal solution reached after {len(attempts)} attempts for {order}/{fc:.2f}")
                    break
                elif abs(delta_fc_2) >= abs(delta_fc_1):
                    print(f"solutions diverging after {len(attempts)} for {order}/{fc:.2f}, discarding last [{delta_fc_1:.2f} vs {delta_fc_2:.2f}]")
                    attempts.pop()
                    break
    return attempts[-1]


if __name__ == '__main__':
    start = 50
    fc_list = [start]
    end = 500
    ppo = 6
    i = 1
    last = fc_list[-1]
    while last <= end:
        last = start * (2 ** (len(fc_list) / ppo))
        fc_list.append(last)

    for order in range(2, 9):
        data = {}
        with (open(f"out_{order}.csv", mode='w')) as f:
            print("fs,adj_fs,delta,factor,delay", file=f)
            divisors = []
            for fc in fc_list:
                optimal = solve(order, fc)
                if optimal.actual_fc > 0.0:
                    print(f"{fc:.2f},{optimal.actual_fc:.2f},{optimal.fc_delta:.2f},{optimal.fc_divisor:.6f},{optimal.delay:.6f}", file=f, flush=True)
                    divisors.append(round(optimal.fc_divisor, 4))
            d = np.array(divisors)
            print(f"{order} -- mean: {np.mean(d):.4f} median: {np.median(d):.4f} std: {np.std(d):.4f}")
            plt.hist(d, bins='auto')
