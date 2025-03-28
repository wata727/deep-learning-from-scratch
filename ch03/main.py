from typing import Dict
import numpy as np
import numpy.typing as npt
from upstream.dataset.mnist import load_mnist
import pickle

def step_function(x: npt.NDArray) -> npt.NDArray:
    return np.array(x > 0, dtype=int)

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-x))

def relu(x: npt.NDArray) -> npt.NDArray:
    return np.maximum(0, x)

def identity_function(x: npt.NDArray) -> npt.NDArray:
    return x

def softmax(a: npt.NDArray) -> npt.NDArray:
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network() -> Dict[str, npt.NDArray]:
    with open("upstream/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network: Dict[str, npt.NDArray], x: npt.NDArray) -> npt.NDArray:
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
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accurancy:" + str(float(accuracy_cnt) / len(x)))
