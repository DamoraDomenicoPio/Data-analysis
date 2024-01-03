import random
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.special


class LogisticBinaryRegression():

    def __init__(self, nMC=1000, m=[1,1], step=0.001):
        self._nMC = nMC
        self._m = m
        self._n_dimension = len(m)
        self._labels = [-1, 1]
        self._alphas = np.arange(0, 1+step, step).tolist()
        self._gammas = self._gamma_calculate(self._alphas)
        self._TPR = list(range(len(self._alphas)))
        self._FPR = list(range(len(self._alphas)))
        self._one_minus_betas = self._one_minus_beta_calculate()
        

    def run(self):
        TP = [0] * len(self._alphas)
        TN = [0] * len(self._alphas)
        FP = [0] * len(self._alphas)
        FN = [0] * len(self._alphas)
        for run in range(self._nMC):
            y, x = self._generate_data()
            for i in range(len(self._alphas)):
                
                if y == -1:
                    if self._dot_product(x, self._m) < self._gammas[i]:
                        TN[i] = TN[i] + 1
                    else:
                        FP[i] = FP[i] + 1
                else:
                    if self._dot_product(x, self._m) > self._gammas[i]:
                        TP[i] = TP[i] + 1
                    else:
                        FN[i] = FN[i] + 1

        for i in range(len(self._alphas)):
            self._FPR[i] =  FP[i] / (FP[i] + TN[i])

            self._TPR[i] = TP[i] / (TP[i] + FN[i])

    def plot_results(self):
        plt.plot(self._FPR, self._TPR, label="estimated ROC")
        plt.plot(self._alphas, self._one_minus_betas, label="true ROC")
        plt.title("Curva ROC")
        plt.show()
    
    def _dot_product(self, x1, x2):
        value = 0
        for i in range(len(x1)):
            value = value + x1[i]*x2[i]
        return value


    def _generate_data(self):
        label = random.choice(self._labels)
        x = list(range(self._n_dimension))
        if label == 1:
            m = self._m
        else:
            m = [0] * self._n_dimension
        for i in range(self._n_dimension):
            x[i] = np.random.normal(m[i], 1)
        return label, x
    

    def _gamma_calculate(self, alphas):
        gammas = list(range(len(alphas)))
        norm_m = self._norm(self._m)
        const = (norm_m**2)/2
        for i in range(len(alphas)):
            gammas[i] = self._inverse_Q_function(alphas[i])*norm_m + const
        return gammas
    
    def _one_minus_beta_calculate(self):
        betas = list(range(len(self._alphas)))
        for i in range(len(self._alphas)):
            betas[i] = self._Q_function((self._inverse_Q_function(self._alphas[i]) - self._norm(self._m)))
        return betas

    def _inverse_Q_function(self, probability):
        return norm.ppf(1 - probability)
    
    def _Q_function(self, x):
        return 0.5 * scipy.special.erfc(x / (2**0.5))

    def _norm(self, x):
        return math.sqrt(sum(x))