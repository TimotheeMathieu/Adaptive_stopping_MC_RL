import numpy as np
import matplotlib.pyplot as plt


for n in range(2, 6):
    for k in range(1, 10):
        decisions = np.load("h2g2/decision_n{}_k{}.npy".format(n,k))
        comparisons = np.load("h2g2/comparison_n{}_k{}.npy".format(n,k))
        hists = []
        for i, deci in enumerate(decisions):
            if deci:
                hists.append(comparisons[i])
        plt.hist(hists)
        plt.show()
