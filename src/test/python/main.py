import numpy as np

# from model.iir import LowShelf
#
# for i in range(50, 2000):
#     q = i / 100.0
#     shelf = LowShelf(48000, 40, q, 10)
#     tf = shelf.getTransferFunction().getMagnitude()
#     lowerF = None
#     lowerG = None
#     upperF = None
#     upperG = None
#     for index, x in np.ndenumerate(tf.y):
#         if x < 6.0 and lowerF is None:
#             lowerF = index
#             lowerG = x
#         if x < 4.0:
#             upperF = index
#             upperG = x
#         if lowerF is not None and upperF is not None:
#             break
#     lowerFreq = tf.x[lowerF]
#     upperFreq = tf.x[upperF]
#     print(f"{q},{shelf.q_to_s()},{str((lowerG - upperG) / np.math.log2(upperFreq/lowerFreq))}")


import matplotlib.style as style


print(style.library.keys())
