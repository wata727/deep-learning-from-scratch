import numpy as np
import numpy.typing as npt
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers: OrderedDict[str, Layer] = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: npt.NDArray, t: npt.NDArray) -> float:
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: npt.NDArray, t: npt.NDArray) -> dict[str, npt.NDArray]:
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x: npt.NDArray, t: npt.NDArray) -> dict[str, npt.NDArray]:
        self.loss(x, t)

        dout = self.lastLayer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW # type: ignore
        grads['b1'] = self.layers['Affine1'].db # type: ignore
        grads['W2'] = self.layers['Affine2'].dW # type: ignore
        grads['b2'] = self.layers['Affine2'].db # type: ignore

        return grads

class MultiLayerNet:
    def __init__(self, input_size: int, hidden_size_list: list[int], output_size: int, weight_init_std: float=0.01, use_batchnorm: bool=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_batchnorm = use_batchnorm
        self.params: dict[str, npt.NDArray] = {}

        self.__init_weight(weight_init_std)

        self.layers: OrderedDict[str, Layer] = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            if use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            self.layers['Relu' + str(idx)] = Relu()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.lastLayer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std: float):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            self.params['W' + str(idx)] = weight_init_std * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: npt.NDArray, t: npt.NDArray) -> float:
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: npt.NDArray, t: npt.NDArray) -> dict[str, npt.NDArray]:
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x: npt.NDArray, t: npt.NDArray) -> dict[str, npt.NDArray]:
        self.loss(x, t)

        dout = self.lastLayer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW # type: ignore
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db # type: ignore
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma # type: ignore
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta # type: ignore

        return grads
