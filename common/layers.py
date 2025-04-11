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

class BatchNormalization(Layer):
    def __init__(self, gamma: npt.NDArray, beta: npt.NDArray, momentum: float = 0.9, running_mean: npt.NDArray | None = None, running_var: npt.NDArray | None = None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x)

        return out.reshape(*self.input_shape)

    def __forward(self, x: npt.NDArray) -> npt.NDArray:
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + 10e-7)
        xn = xc / std

        self.batch_size = x.shape[0]
        self.xc = xc
        self.xn = xn
        self.std = std
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        if dout.ndim != 2:
            N, C, H, W = self.input_shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout: npt.NDArray) -> npt.NDArray:
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class Dropout(Layer):
    def __init__(self, dropout_ratio: float = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: npt.NDArray, train_flg: bool = True) -> npt.NDArray:
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        return dout * self.mask
