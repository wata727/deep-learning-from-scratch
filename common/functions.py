import numpy as np
import numpy.typing as npt

def sigmoid(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    return 1 / (1 + np.exp(-x))

def softmax(a: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    c = np.max(a, axis=-1, keepdims=True)
    exp_a: npt.NDArray[np.double] = np.exp(a - c)
    sum_exp_a: npt.NDArray[np.double] = np.sum(exp_a, axis=-1, keepdims=True)
    return exp_a / sum_exp_a

def cross_entropy_error(y: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    v: float = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return v
