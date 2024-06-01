import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def generate_data(center1, center2, n_samples):
    X = np.vstack((np.random.randn(n_samples, 2) + center1, np.random.randn(n_samples, 2) + center2))
    Y = np.hstack((np.ones(n_samples), -np.ones(n_samples)))
    rand_indices = np.random.permutation(2 * n_samples)
    X = X[rand_indices]
    Y = Y[rand_indices]
    return X, Y


@jit(nopython=True)
def train_svm(X, Y, learning_rate=0.00001, C=1, num_iters=10000):
    n = X.shape[0]
    alpha = np.zeros(n)
    lamda = 0
    beita = 1
    Xt = X * Y[:, np.newaxis]
    one = np.ones(n)

    for k in range(num_iters):
        grad = Xt @ Xt.T @ alpha - one + lamda * Y + beita * (Y.T @ alpha) * Y
        alpha = alpha - learning_rate * grad
        alpha = np.clip(alpha, 0, C)
        lamda = lamda + beita * (Y.T @ alpha)

    w = Xt.T @ alpha
    support_vectors = np.where((alpha > 0) & (alpha < C))[0]
    b = np.mean(Y[support_vectors] - X[support_vectors] @ w)

    return w, b, support_vectors


def plot_decision_boundary(dataX, dataY, w, b, support_vectors):
    x1 = np.linspace(-2, 7, 1000)
    y1 = (-b - w[0] * x1) / w[1]
    y2 = (1 - b - w[0] * x1) / w[1]
    y3 = (-1 - b - w[0] * x1) / w[1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dataX[dataY == 1][:, 0], dataX[dataY == 1][:, 1], color='g', marker='o', label='class 1')
    ax.scatter(dataX[dataY == -1][:, 0], dataX[dataY == -1][:, 1], color='b', marker='*', label='class 2')
    ax.plot(x1, y1, 'k', label='classification surface')
    ax.plot(x1, y2, 'k--', label='boundary')
    ax.plot(x1, y3, 'k--', label='boundary')
    for sv in support_vectors:
        if dataY[sv] == 1:
            ax.scatter(dataX[sv, 0], dataX[sv, 1],
                       facecolors='none', edgecolors='g', marker='o', s=100,
                       label='support vector class 1' if sv == support_vectors[0] else "")
        else:
            ax.scatter(dataX[sv, 0], dataX[sv, 1],
                       facecolors='none', edgecolors='b', marker='o', s=100,
                       label='support vector class 2' if sv == support_vectors[0] else "")
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.legend()
    return fig



