import numpy as np
from compare_agents import Two_AgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


combs = [[3, 7], [4, 5], [4, 4], [2, 10]]
for comb in combs:
    for i in range(2):
        n, k = comb[i], comb[1-i]
        for exps in range(100):
            filename = "h2g2/n-{}-k-{}-rep-{}.pkl".format(n,k,exps)
            comparator = Two_AgentsComparator(B = 100_000, K = k,n = n)
            with open(filename, "rb") as f:
                comparator.__dict__ = pickle.load(f)
            print(comparator.decision, n*k, comparator.n_iter/2)
