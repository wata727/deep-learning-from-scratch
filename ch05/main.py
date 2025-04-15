import numpy as np
import numpy.typing as npt
from collections import OrderedDict
from typing import TypedDict, cast
from abc import ABC, abstractmethod
from upstream.dataset.mnist import load_mnist
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
from common.types import *

class MulLayer:
    x: float
    y: float

    def __init__(self) -> None:
        pass

    def forward(self, x: float, y: float) -> float:
        self.x = x
        self.y = y
        out = x * y

        return out

    def backword(self, dout: float) -> tuple[float, float]:
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x: float, y: float) -> float:
        out = x + y
        return out

    def backword(self, dout: float) -> tuple[float, float]:
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class Layer(ABC):
    @abstractmethod
    def forward(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        pass
    @abstractmethod
    def backward(self, dout: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        pass

class Relu(Layer):
    mask : npt.NDArray[np.bool]

    def __init__(self) -> None:
        pass

    def forward(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid(Layer):
    out: npt.NDArray[np.double]

    def __init__(self) -> None:
        pass

    def forward(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backword(self, dout: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine(Layer):
    W: npt.NDArray[np.double]
    b: npt.NDArray[np.double]
    x: npt.NDArray[np.double]
    dW: npt.NDArray[np.double]
    db: npt.NDArray[np.double]

    def __init__(self, W: npt.NDArray[np.double], b: npt.NDArray[np.double]) -> None:
        self.W = W
        self.b = b

    def forward(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        self.x = x
        out: npt.NDArray[np.double] = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        dx: npt.NDArray[np.double] = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    loss: float
    y: npt.NDArray[np.double]
    t: npt.NDArray[np.double]

    def __init__(self) -> None:
        pass

    def forward(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout: float = 1) -> npt.NDArray[np.double]:
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

Params = TypedDict('Params', {
    'W1': Matrix[np.double],
    'b1': Vector[np.double],
    'W2': Matrix[np.double],
    'b2': Vector[np.double]
})

class TwoLayerNet:
    params: Params
    layers: OrderedDict[str, Layer]

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
        self.params = {
            'W1': cast(Matrix[np.double], weight_init_std * np.random.randn(input_size, hidden_size)),
            'b1': np.zeros(hidden_size),
            'W2': cast(Matrix[np.double], weight_init_std * np.random.randn(hidden_size, output_size)),
            'b2': np.zeros(output_size)
        }

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

    def accuracy(self, x: Matrix[np.double], t: Matrix[np.double]) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy: float = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> Params:
        loss_W = lambda W: self.loss(x, t)

        return {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2']),
        }

    def gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> Params:
        self.loss(x, t)

        dout = self.lastLayer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        return {
            'W1': cast(Matrix[np.double], cast(Affine, self.layers['Affine1']).dW),
            'b1': cast(Vector[np.double], cast(Affine, self.layers['Affine1']).db),
            'W2': cast(Matrix[np.double], cast(Affine, self.layers['Affine2']).dW),
            'b2': cast(Vector[np.double], cast(Affine, self.layers['Affine2']).db),
        }

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # type: ignore[no-untyped-call]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

diff = np.average(np.abs(grad_backprop['W1'] - grad_numerical['W1']))
print("W1:" + str(diff))
diff = np.average(np.abs(grad_backprop['b1'] - grad_numerical['b1']))
print("b1:" + str(diff))
diff = np.average(np.abs(grad_backprop['W2'] - grad_numerical['W2']))
print("W2:" + str(diff))
diff = np.average(np.abs(grad_backprop['b2'] - grad_numerical['b2']))
print("b2:" + str(diff))

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    network.params['W1'] -= learning_rate * grad['W1']
    network.params['b1'] -= learning_rate * grad['b1']
    network.params['W2'] -= learning_rate * grad['W2']
    network.params['b2'] -= learning_rate * grad['b2']

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
