import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import cast
from common.layers import *
from common.gradient import numerical_gradient

class Network(ABC):
    params: dict[str, npt.NDArray[np.double]]

    @abstractmethod
    def gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> dict[str, npt.NDArray[np.double]]:
        pass
    @abstractmethod
    def loss(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        pass
    @abstractmethod
    def accuracy(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        pass

class TwoLayerNet(Network):
    params: dict[str, npt.NDArray[np.double]]

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

    def predict(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy: float = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> dict[str, npt.NDArray[np.double]]:
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> dict[str, npt.NDArray[np.double]]:
        self.loss(x, t)

        dout = self.lastLayer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = cast(Affine, self.layers['Affine1']).dW
        grads['b1'] = cast(Affine, self.layers['Affine1']).db
        grads['W2'] = cast(Affine, self.layers['Affine2']).dW
        grads['b2'] = cast(Affine, self.layers['Affine2']).db

        return grads

class MultiLayerNet(Network):
    def __init__(self, input_size: int, hidden_size_list: list[int], output_size: int, weight_init_std: str | float = 'relu', use_batchnorm: bool=False, weight_decay_lambda: float=0.0, use_dropout: bool=False, dropout_ration: float=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.params: dict[str, npt.NDArray[np.double]] = {}

        self.__init_weight(weight_init_std)

        self.layers: OrderedDict[str, Layer] = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            if use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            self.layers['Relu' + str(idx)] = Relu()
            if use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.lastLayer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std: str | float) -> None:
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            assert isinstance(scale, float)

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x: npt.NDArray[np.double], train_flg: bool = False) -> npt.NDArray[np.double]:
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = cast(Dropout, layer).forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double], train_flg: bool = False) -> float:
        y = self.predict(x, train_flg)

        weight_decay = np.float64(0.0)
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.lastLayer.forward(y, t) + weight_decay

    def accuracy(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy: float = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> dict[str, npt.NDArray[np.double]]:
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> dict[str, npt.NDArray[np.double]]:
        self.loss(x, t, train_flg=True)

        dout = self.lastLayer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W # type: ignore
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db # type: ignore
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma # type: ignore
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta # type: ignore

        return grads
