import numpy as np
import numpy.typing as npt
from common.networks import Network
from common.optimizer import Optimizer

class Trainer:
    def __init__(self, network: Network, x_train: npt.NDArray[np.double], t_train: npt.NDArray[np.double], x_test: npt.NDArray[np.double], t_test: npt.NDArray[np.double],
                 optimizer: Optimizer, epochs: int = 20, mini_batch_size: int = 100, verbose: bool = True):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.verbose = verbose

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.train_loss_list: list[float] = []
        self.train_acc_list: list[float] = []
        self.test_acc_list: list[float] = []

    def train(self) -> tuple[float, float]:
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
                if self.verbose:
                    print(train_acc, test_acc)

        return train_acc, test_acc
