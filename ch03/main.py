import numpy as np
import numpy.typing as npt
from typing import TypedDict
from upstream.dataset.mnist import load_mnist
import pickle

def step_function(x: npt.NDArray[np.double]) -> npt.NDArray[np.int_]:
    return np.array(x > 0, dtype=int)

def sigmoid(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    return 1 / (1 + np.exp(-x))

def relu(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    return np.maximum(0, x)

def identity_function(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    return x

def softmax(a: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    c = np.max(a)
    exp_a: npt.NDArray[np.double] = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

type Vector[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]
type Matrix[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]

def get_data() -> tuple[Matrix[np.double], Vector[np.int_]]:
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False) # type: ignore[no-untyped-call]
    return x_test, t_test

Network = TypedDict('Network', {
    'W1': Matrix[np.double],
    'b1': Vector[np.double],
    'W2': Matrix[np.double],
    'b2': Vector[np.double],
    'W3': Matrix[np.double],
    'b3': Vector[np.double]
})

def init_network() -> Network:
    with open("upstream/ch03/sample_weight.pkl", 'rb') as f:
        network: Network = pickle.load(f)

    return network

def predict(network: Network, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p: Vector[np.int_] = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accurancy:" + str(float(accuracy_cnt) / len(x)))
