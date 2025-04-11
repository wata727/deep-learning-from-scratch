import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from common.functions import softmax, cross_entropy_error
from upstream.common.util import im2col, col2im

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
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
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
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

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

class Convolution:
    def __init__(self, W: npt.NDArray, b: npt.NDArray, stride: int = 1, pad: int = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling(Layer):
    def __init__(self, pool_h: int, pool_w: int, stride: int = 2, pad: int = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout: npt.NDArray) -> npt.NDArray:
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
