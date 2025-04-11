import numpy as np
from upstream.dataset.mnist import load_mnist
from common.utils import shuffle_dataset
from common.optimizer import SGD
from common.networks import MultiLayerNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

optimization_trial = 100
for _ in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    optimizer = SGD(lr=lr)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val, epochs=50, mini_batch_size=100, optimizer=optimizer, verbose=False)
    train_acc, test_acc = trainer.train()
    print(f"weight_decay: {weight_decay}, lr: {lr}, train_acc: {train_acc}, test_acc: {test_acc}")

# weight_decay: 5.199054482411417e-05, lr: 0.008091269637677583, train_acc: 0.8975, test_acc: 0.82
# weight_decay: 2.818730117420757e-06, lr: 0.009620917000403083, train_acc: 0.91, test_acc: 0.83
# weight_decay: 4.202453284274309e-05, lr: 0.0066378282443620325, train_acc: 0.8725, test_acc: 0.8
# weight_decay: 4.059856565515096e-08, lr: 0.0054812354008213215, train_acc: 0.825, test_acc: 0.73
# weight_decay: 3.5144754021021166e-05, lr: 0.006061530460134854, train_acc: 0.82, test_acc: 0.73
# weight_decay: 1.367707726460586e-07, lr: 0.008085303463731583, train_acc: 0.8775, test_acc: 0.78
# weight_decay: 3.759499210641065e-08, lr: 0.005036279948946462, train_acc: 0.845, test_acc: 0.74
