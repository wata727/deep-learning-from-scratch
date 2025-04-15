from upstream.dataset.mnist import load_mnist
from common.networks import MultiLayerNet
from common.optimizer import SGD
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # type: ignore[no-untyped-call]
x_train = x_train[:300]
t_train = t_train[:300]

# use_dropout = False # 1.0 0.7619
use_dropout = True # 0.8133333333333334 0.6453
dropout_ration = 0.15
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ration)
optimizer = SGD(lr=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test, optimizer=optimizer, epochs=301, mini_batch_size=100)
trainer.train()
