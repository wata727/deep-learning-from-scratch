from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from typing import TypedDict, cast
from upstream.dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from common.types import *

def sum_squared_error(y: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    v: float = -np.sum(t * np.log(y + 1e-7)) / batch_size
    return v

def numerical_gradient[T: tuple[int, ...]](f: Callable[[npt.NDArray[np.double]], float], x: np.ndarray[T, np.dtype[np.double]]) -> np.ndarray[T, np.dtype[np.double]]:
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=[['readwrite']])
    while not it.finished:
        idx = it.multi_index
        v = x[idx]
        x[idx] = v + h
        fxh1 = f(x)

        x[idx] = v - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = v
        it.iternext()

    return grad

Params = TypedDict('Params', {
    'W1': Matrix[np.double],
    'b1': Vector[np.double],
    'W2': Matrix[np.double],
    'b2': Vector[np.double]
})

class TwoLayerNet:
    params: Params

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
        self.params = {
            'W1': cast(Matrix[np.double], weight_init_std * np.random.randn(input_size, hidden_size)),
            'b1': np.zeros(hidden_size),
            'W2': cast(Matrix[np.double], weight_init_std * np.random.randn(hidden_size, output_size)),
            'b2': np.zeros(output_size)
        }

    def predict(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x: Matrix[np.double], t: Matrix[np.double]) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accurancy: float = np.sum(y == t) / float(x.shape[0])
        return accurancy

    def numerical_gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> Params:
        loss_W = lambda W: self.loss(x, t)

        return {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2'])
        }

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # type: ignore[no-untyped-call]

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    network.params['W1'] -= cast(Matrix[np.double], learning_rate * grad['W1'])
    network.params['b1'] -= cast(Vector[np.double], learning_rate * grad['b1'])
    network.params['W2'] -= cast(Matrix[np.double], learning_rate * grad['W2'])
    network.params['b2'] -= cast(Vector[np.double], learning_rate * grad['b2'])

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
