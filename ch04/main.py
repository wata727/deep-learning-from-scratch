from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from upstream.dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def sum_squared_error(y: npt.NDArray, t: npt.NDArray) -> float:
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y: npt.NDArray, t: npt.NDArray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f: Callable, x: npt.NDArray) -> npt.NDArray:
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

class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x: npt.NDArray, t: npt.NDArray) -> float:
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x: npt.NDArray, t: npt.NDArray) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accurancy = np.sum(y == t) / float(x.shape[0])
        return accurancy

    def numerical_gradient(self, x: npt.NDArray, t: npt.NDArray) -> dict[str, npt.NDArray]:
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

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

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
