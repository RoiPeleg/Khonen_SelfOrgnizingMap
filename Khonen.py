import numpy as np
import math
import matplotlib.pyplot as plt


class Khonen:
    def __init__(self, m: int, n: int, input_size: int, iterations: int):
        self.m = m
        self.n = n
        self.learning_rate = 0.2
        self.weights = np.array([np.random.uniform(-2, 2, (m, n)) for i in range(input_size)]).astype(np.float64)
        self.iterations = iterations

    def getCorr(self):
        return self.weights

    def fit(self, data: np.ndarray):
        r = 2
        ws = self.weights.copy()
        for it in range(self.iterations):
            for x in data:
                minerror, index = math.inf, (-1, -1)
                for i in range(self.m):
                    for j in range(self.n):
                        error = ((x - self.weights[:, i, j]) ** 2).sum()
                        (minerror, index) = (error, (i, j)) if error < minerror else (minerror, index)
                w = self.weights[:, min(index[0]-r,0):max(self.m,index[1]+r), min(index[1]-r,0):max(self.n,index[1]+r)]
                print(np.subtract(w,x).shape)
                w += self.learning_rate * (np.subtract(w,x.reshape(2,1)))
            self.learning_rate *= 0.99
            r = (max(2, r - 1))



length = np.sqrt(np.random.uniform(0, 2, 1000))
angle = np.pi * np.random.uniform(0, 2, 1000)

x = length * np.cos(angle)
y = length * np.sin(angle)
circle = np.vstack((x, y)).T
k = Khonen(30, 1, 2, 150)
w = k.getCorr()
k.fit(circle)
plt.scatter(x, y, c='r')
plt.plot(k.getCorr()[0], k.getCorr()[1], marker='o')
plt.show()
