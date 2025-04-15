import numpy as np
from upstream.dataset.mnist import load_mnist
from common.networks import MultiLayerNet
from common.optimizer import SGD
from common.layers import Affine

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # type: ignore[no-untyped-call]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10)

all_size_list = [784, 100, 100, 100, 100, 10]
for idx in range(1, len(all_size_list)):
    # std=0.01: 0.11236666666666667 0.1135
    # network.params['W' + str(idx)] = 0.01 * np.random.randn(all_size_list[idx-1], all_size_list[idx])

    # He initialization: 0.9984333333333333 0.978
    network.params['W' + str(idx)] = np.sqrt(2.0 / all_size_list[idx-1]) * np.random.randn(all_size_list[idx-1], all_size_list[idx])

    network.params['b' + str(idx)] = np.zeros(all_size_list[idx])
    network.layers['Affine' + str(idx)] = Affine(network.params['W' + str(idx)], network.params['b' + str(idx)])

optimizer = SGD(lr=0.1)

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
