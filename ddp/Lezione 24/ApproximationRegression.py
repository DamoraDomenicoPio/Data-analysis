import numpy as np
import matplotlib.pyplot as plt
import math

class ApproximationRegression():

    def __init__(self, nMC=100, N=1000000, a=3):
        self._nMC = nMC
        self._N = N
        self._a = a
        law = math.sqrt(N)
        self._kn = math.floor(law)
        self._h = 1/law

    def run(self):
        y, x = self._generate_data()
        y_kNN, x_KNN = self.run_knn(y, x)
        y_NK, x_NK = self.run_NK(y, x)
        #plt.plot(x, y)
        plt.plot(x_NK, y_NK, label="NK")
        plt.plot(x_KNN, y_kNN, label="KNN")

        x = np.arange(0, self._a, 0.01)
        y = np.sin(2 * np.pi * x)
        plt.plot(x, y, label="true")

        plt.legend()

        plt.show()


    #aggiusta i bordi
    # peova a fare x_KNN[k] = x[i+kn/2]
    def run_knn(self, y=None, x=None):
        if y == None:
            y, x = self._generate_data()
        x_KNN = list(range(math.floor(self._N/self._kn)))
        y_KNN = list(range(math.floor(self._N/self._kn)))
        i = 0
        k = 0
        while i < self._N:
            x_KNN[k] = x[i]
            values = y[i:i+self._kn]
            i = i+self._kn + 1
            y_KNN[k] = self._mean(values)
            k = k + 1

        return y_KNN, x_KNN


    def run_NK(self, y, x):
        if y == None:
            y, x = self._generate_data()
        n_intervals = math.floor(self._a/self._h)
        x_NK = list(range(n_intervals))
        y_NK = list(range(n_intervals))
        k = 0
        for i in range(n_intervals):
            x_NK[i] = self._h*(1/2 + i)
            j = 0
            values = 0
            while x[k] <= self._h*(i+1):
                values = values + y[k]
                j = j + 1
                k = k + 1
                if k >= self._N:
                    break
            y_NK[i] = values/j

        return y_NK, x_NK
            

        
    def _generate_data(self):
        x = np.random.uniform(0, self._a, self._N).tolist()
        x.sort()
        errors = np.random.randn(self._N)
        y = list(range(self._N))
        for i in range(0, len(x)):
            y[i] = np.sin(2 * np.pi * x[i]) + errors[i]
        return y, x

    def _mean(self, data):
        if not isinstance(data, list):
            return data
        values = 0
        for i in range(0, len(data)):
            values = values + data[i]
        return values/len(data)
