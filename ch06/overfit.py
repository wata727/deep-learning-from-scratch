import numpy as np
from upstream.dataset.mnist import load_mnist
from common.networks import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # type: ignore[no-untyped-call]
x_train = x_train[:300]
t_train = t_train[:300]

# 0.9966666666666667 0.7662
# network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10)

# 0.91 0.7268
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10, weight_decay_lambda=0.1)

optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
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

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
