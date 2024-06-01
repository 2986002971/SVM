import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 数据生成函数
def gaussian_generate(center1, center2, num):
    data_x = np.empty((0, 2))
    data_y = np.empty(0)

    for _ in range(num):
        if np.random.random() < 0.5:
            x = center1 + np.random.randn(2)
            y = 1
        else:
            x = center2 + np.random.randn(2)
            y = -1

        data_x = np.vstack((data_x, x))
        data_y = np.hstack((data_y, y))

    return data_x, data_y

# 生成数据
center1 = np.array([1, 1])
center2 = np.array([3, 3])
num_samples = 100
data_x, data_y = gaussian_generate(center1, center2, num_samples)

# 使用sklearn的SVM实现
clf = SVC(kernel='linear', C=1.0)
clf.fit(data_x, data_y)

w = clf.coef_[0]
b = clf.intercept_[0]
support_vectors = clf.support_

print("w:", w)
print("b:", b)
print("Support Vectors:", support_vectors)

# 绘制SVM分类结果
def plot_svm(X, y, w, b, support_vectors):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    xx = np.linspace(x_min, x_max, 100)
    yy = (-w[0] * xx - b) / w[1]

    plt.plot(xx, yy, 'k-', label='Decision Boundary')
    plt.fill_between(xx, yy - 1/w[1], yy + 1/w[1], color='gray', alpha=0.2, label='Margin')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM with sklearn')
    plt.show()

plot_svm(data_x, data_y, w, b, support_vectors)
