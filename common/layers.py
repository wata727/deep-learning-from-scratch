import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from common.functions import softmax, cross_entropy_error

class Layer(ABC):
    @abstractmethod
    def forward(self, x: npt.NDArray) -> npt.NDArray:
        pass
    @abstractmethod
    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        pass

class Relu(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backword(self, dout: npt.NDArray) -> npt.NDArray:
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine(Layer):
    def __init__(self, W: npt.NDArray, b: npt.NDArray):
        self.W = W
        self.b = b
        self.x: npt.NDArray = np.empty(0)
        self.dW = None
        self.db = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: npt.NDArray, t: npt.NDArray) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout: float = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
