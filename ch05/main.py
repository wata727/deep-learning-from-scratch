import numpy as np
import numpy.typing as npt
from collections import OrderedDict
from abc import ABC, abstractmethod
from upstream.dataset.mnist import load_mnist
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

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
    def __init__(self):
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

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

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

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
