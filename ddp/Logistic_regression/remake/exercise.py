from LogisticBinaryRegression import LogisticBinaryRegression
from SGD import SGD

### LOGISTIC BINARY REGRESSION ###
# l = LogisticBinaryRegression(nMC=1000, m=[10, 10, 10])
# l.run()
# l.plot_results()

### SGD ###
s = SGD(nMC=100000, m=[0.5, 0.5], step_stop=10)
s.run()
s.plot_costs()
s.test_beta()