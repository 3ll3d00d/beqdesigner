import matplotlib.pyplot as plt
import numpy as np

from model.iir import ComplexLowPass, FilterType
from model.jriver.filter import optimise_mds

f = plt.figure()
plt.grid(visible=True)


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
                optimal = optimise_mds(order, fc)
                if optimal.actual_fc > 0.0:
                    print(f"{fc:.2f},{optimal.actual_fc:.2f},{optimal.fc_delta:.2f},{optimal.fc_divisor:.6f},{optimal.delay:.6f}", file=f, flush=True)
                    divisors.append(round(optimal.fc_divisor, 4))
            d = np.array(divisors)
            print(f"{order} -- mean: {np.mean(d):.4f} median: {np.median(d):.4f} std: {np.std(d):.4f}")
            plt.hist(d, bins='auto')
            plt.show()
