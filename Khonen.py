import numpy as np
import math
import matplotlib.pyplot as plt


class Khonen:
    def __init__(self, m: int, n: int, input_size: int, iterations: int):
        self.m = m
        self.n = n
        self.learning_rate = 0.2
        self.weights = np.array([np.random.uniform(-2, 2, (m, n)) for i in range(input_size)])
        self.iterations = iterations

    def getCorr(self):
        return self.weights

    def fit(self, data: np.ndarray):
        r = 9
        for it in range(self.iterations):
            minerror, index = math.inf, (-1, -1)
            for x in data:
                for i in range(self.n - 1):
                    for j in range(self.m - 1):
                        error = ((x - self.weights[:, i, j]) ** 2).sum()
                        (minerror, index) = (error, (i, j)) if error < minerror else (minerror, index)
                self.weights[:, max(0, index[0] - r):min(index[0] + r + 1, self.n),
                max(index[1] - r, 0):min(index[1] + r + 1, self.m)] += self.learning_rate * minerror
        self.learning_rate *= 0.9
        r = (max(1, r - 1))


length = np.sqrt(np.random.uniform(-2, 2, 1000))
angle = np.pi * np.random.uniform(0, 2, 1000)

x = length * np.cos(angle)
y = length * np.sin(angle)
print(x.shape)
circle = np.vstack((x, y)).T
print(circle.shape)
k = Khonen(30, 1, 2, 1000)
k.fit(circle)
plt.scatter(x, y, c='r')
plt.plot(k.getCorr()[0], k.getCorr()[1], marker='o')
plt.show()
