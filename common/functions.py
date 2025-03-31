import numpy as np
import numpy.typing as npt

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-x))

def softmax(a: npt.NDArray) -> npt.NDArray:
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a
