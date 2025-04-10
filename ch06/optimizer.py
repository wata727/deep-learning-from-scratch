import numpy as np
import numpy.typing as npt
from upstream.dataset.mnist import load_mnist
from common.networks import TwoLayerNet

class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: dict[str, npt.NDArray], grads: dict[str, npt.NDArray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v: dict[str, npt.NDArray] | None = None

    def update(self, params: dict[str, npt.NDArray], grads: dict[str, npt.NDArray]):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.h: dict[str, npt.NDArray] | None = None

    def update(self, params: dict[str, npt.NDArray], grads: dict[str, npt.NDArray]):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) 

class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m: dict[str, npt.NDArray] | None = None
        self.v: dict[str, npt.NDArray] | None = None

    def update(self, params: dict[str, npt.NDArray], grads: dict[str, npt.NDArray]):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        assert self.m is not None
        assert self.v is not None

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# optimizer = SGD(lr=0.1) # 0.9787833333333333 0.9698
# optimizer = Momentum(lr=0.1) # 0.9914 0.9721
# optimizer = AdaGrad(lr=0.1) # 0.986 0.968
optimizer = Adam() # 0.9864166666666667 0.9707

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
