from LogisticBinaryRegression import LogisticBinaryRegression
from SGD import SGD

### LOGISTIC BINARY REGRESSION ###
# l = LogisticBinaryRegression(nMC=1000, m=[10, 10, 10])
# l.run()
# l.plot_results()

### SGD ###
s = SGD(nMC=1000, m=[1, 1], step_stop=5)
s.run()
s.plot_costs()
s.test_beta()
