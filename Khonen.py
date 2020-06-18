import numpy as np
import math
import matplotlib.pyplot as plt

class Khonen:
    def __init__(self, m: int, n: int, input_size: int, iterations: int):
        self.m = m
        self.n = n
        self.learning_rate = 0.3
        self.weights = self.setLine()
        self.iterations = iterations

    def getCorr(self):
        return self.weights

    def setLine(self):
        w = np.zeros((2, 30, 1))
        w[0] = np.random.uniform(-2, 2, (30, 1))
        w[1] = np.ones((30, 1))
        return w

    def setCricle(self):
        pass

    def fit(self, data: np.ndarray):
        r0 = 8
        r = r0
        lr = self.learning_rate
        for it in range(self.iterations):
            for x in data:
                minerror, index = math.inf, (-1, -1)
                for i in range(self.m):
                    for j in range(self.n):
                        error = ((x - self.weights[:, i, j]) ** 2).sum()
                        (minerror, index) = (error, (i, j)) if error < minerror else (minerror, index)
                window = self.weights[:, max(index[0] - r, 0):min(index[0] + r, self.m), max(index[1] - r, 0):min(index[1] + r, self.m)]
                window += (self.learning_rate*np.exp(- self.weights[:, index[0], index[1]]/(2*(r**2))).reshape((x.shape[0], 1, self.n))*(np.subtract(x.reshape((x.shape[0], 1, self.n)), window)))
            self.learning_rate = lr *np.exp(-it/self.iterations)
            r = min(int(r0*np.exp(-it//1000)),3)


length = np.sqrt(np.random.uniform(0, 2, 800))
angle = np.pi * np.random.uniform(0, 2, 800)

x = length * np.cos(angle)
y = length * np.sin(angle)
circle = np.vstack((x, y)).T
k = Khonen(30, 1, 2, 600)
k.fit(circle)
plt.scatter(x, y, c='r')
plt.plot(k.getCorr()[0], k.getCorr()[1], marker='o')
plt.show()
