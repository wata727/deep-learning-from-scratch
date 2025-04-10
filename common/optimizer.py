import numpy.typing as npt

class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: dict[str, npt.NDArray], grads: dict[str, npt.NDArray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
