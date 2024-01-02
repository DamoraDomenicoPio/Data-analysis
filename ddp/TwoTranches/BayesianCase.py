import random
import numpy as np
import matplotlib.pyplot as plt
import math

class BayesianCase():

    def __init__(self, nMC=100, N=10, sy=1, sw=1):
        self._nMC = nMC
        self._N = N
        self._sy = sy
        self._sw = sw

        self._MSE_tml = 0
        self._MSE_tbayes = 0

        self._th_err = (sy*sw/N)/(sy+sw/N)


    def run(self):
        est_bayes = list(range(self._nMC))
        est_ml = list(range(self._nMC))
        y = list(range(self._nMC))
        a = self._sy/(self._N*self._sy + self._sw)
        for i in range(0, self._nMC):
            y[i] = np.random.normal(0, math.sqrt(self._sy), 1)[0]
            x = np.random.normal(y[i], math.sqrt(self._sw), self._N)
            sum_x = sum(x)
            est_ml[i] = sum_x/self._N
            est_bayes[i] = sum_x*a

        self._MSE_tml = self._MSE(est_ml, y)
        self._MSE_tbayes = self._MSE(est_bayes, y)
        
    def get_th_err(self):
        return self._th_err
    
    def get_sy(self):
        return self._sy
    
    def get_sw(self):
        return self._sw
    
    def get_N(self):
        return self._N

    def set_sy(self, sy):
        self._sy = sy
        self._th_err = (self._sy*self._sw/self._N)/(self._sy+self._sw/self._N)


    def set_sw(self, sw):
        self._sw = sw
        self._th_err = (self._sy*self._sw/self._N)/(self._sy+self._sw/self._N)

    def set_N(self, N):
        self._N = N
        self._th_err = (self._sy*self._sw/self._N)/(self._sy+self._sw/self._N)

    def results(self):
        return {"MSE_ml" : self._MSE_tml, "MSE_bayes" : self._MSE_tbayes}


    def print_results(self):
        print("*** RESULTS ***")
        print()
        print("Parameters")
        print("runs MonteCarlo:", self._nMC)
        print("N:", self._N, " mean:", 0, " sigmay:", self._sy, " sigmaw:", self._sw)
        print()
        print("MSE ml:", self._MSE_tml)
        print("MSE bayes:", self._MSE_tbayes)
        print()
        print("***************")

    
    def _MSE(self, data, means):
        values = 0
        for i in range(0,len(data)):
            values = values + ((data[i] - means[i])**2)
        return values/len(data)
    

    def plot_results(self, xlable_name, xlable, MSE_tml ,MSE_tbayes, th_err, others=None, others_label=None, others2=None, others_label2=None):
        
        plt.plot(xlable, MSE_tml, label="ML")
        plt.plot(xlable, MSE_tbayes, label="Bayes")
        plt.plot(xlable, th_err, label="th err")
        if others != None:
            plt.plot(xlable, others, label=others_label)
        if others2 != None:
            plt.plot(xlable, others2, label=others_label2)

        plt.xscale('log')

        plt.xlabel(xlable_name)
        plt.ylabel("MSE")

        plt.legend()
        plt.show()