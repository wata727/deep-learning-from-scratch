import numpy as np
import numpy.typing as npt
from common.networks import MultiLayerNet
from common.optimizer import SGD

class Trainer:
    def __init__(self, network: MultiLayerNet, x_train: npt.NDArray, t_train: npt.NDArray, x_test: npt.NDArray, t_test: npt.NDArray,
                 optimizer: SGD, epochs: int = 20, mini_batch_size: int = 100):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = mini_batch_size

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        for i in range(self.max_iter):
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)

            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)

            if i % self.iter_per_epoch == 0:
                train_acc = self.network.accuracy(self.x_train, self.t_train)
                test_acc = self.network.accuracy(self.x_test, self.t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print(train_acc, test_acc)
