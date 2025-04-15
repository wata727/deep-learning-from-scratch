import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, params: dict[str, npt.NDArray[np.double]], grads: dict[str, npt.NDArray[np.double]]) -> None:
        pass

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: dict[str, npt.NDArray[np.double]], grads: dict[str, npt.NDArray[np.double]]) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Adam(Optimizer):
    m: dict[str, npt.NDArray[np.double]]
    v: dict[str, npt.NDArray[np.double]]

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0

    def update(self, params: dict[str, npt.NDArray[np.double]], grads: dict[str, npt.NDArray[np.double]]) -> None:
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
