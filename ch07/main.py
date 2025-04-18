import numpy as np
import numpy.typing as npt
from collections import OrderedDict
from typing import TypedDict, cast
from upstream.dataset.mnist import load_mnist
from common.layers import *
from common.optimizer import Adam
from common.networks import Network
from common.trainer import Trainer

ConvParam = TypedDict('ConvParam', {
    'filter_num': int,
    'filter_size': int,
    'pad': int,
    'stride': int
})

class SimpleConvNet(Network):
    def __init__(self, input_dim: tuple[int, int, int]=(1, 28, 28),
                 conv_param: ConvParam={'filter_num': 30, 'filter_size': 5,
                             'pad': 0, 'stride': 1},
                             hidden_size: int=100, output_size: int=10, weight_init_std: float=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers: OrderedDict[str, Layer] = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> float:
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double], batch_size: int=100) -> float:
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x: npt.NDArray[np.double], t: npt.NDArray[np.double]) -> dict[str, npt.NDArray[np.double]]:
        self.loss(x, t)

        dout = self.last_layer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = cast(Convolution, self.layers['Conv1']).dW
        grads['b1'] = cast(Convolution, self.layers['Conv1']).db
        grads['W2'] = cast(Affine, self.layers['Affine1']).dW
        grads['b2'] = cast(Affine, self.layers['Affine1']).db
        grads['W3'] = cast(Affine, self.layers['Affine2']).dW
        grads['b3'] = cast(Affine, self.layers['Affine2']).db

        return grads

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False) # type: ignore[no-untyped-call]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = Adam(lr=0.001)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer=optimizer)
trainer.train() # 0.99925 0.9892
