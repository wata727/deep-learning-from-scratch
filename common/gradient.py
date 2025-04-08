import numpy as np
import numpy.typing as npt
from typing import Callable

def numerical_gradient(f: Callable, x: npt.NDArray) -> npt.NDArray:
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=[['readwrite']])
    while not it.finished:
        idx = it.multi_index
        v = x[idx]
        x[idx] = v + h
        fxh1 = f(x)

        x[idx] = v - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = v
        it.iternext()

    return grad
