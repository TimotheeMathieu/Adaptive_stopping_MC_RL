import numpy as np
from compare_agents import Two_AgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ks = []
scalar_list1 = np.loadtxt("h2g2/evals_sac_hc.txt")
scalar_list2 = np.loadtxt("h2g2/evals_td3_hc.txt")
# sns.set_style("whitegrid")
# sns.kdeplot(scalar_list1, x = "evaluation means", label = "SAC")
# sns.kdeplot(scalar_list2, x = "evaluation means", label  = "TD3")
# plt.xlabel("evaluations means")
# plt.legend()
# plt.savefig("h2g2/sac_vs_td3_hc.png")
# all_comparator_stuff = []
combs = [[3, 7], [4, 5], [4, 4], [2, 10]]
for comb in combs:
    for i in range(2):
        n, k = comb[i], comb[1-i]
        for exps in range(100):
            filename = "h2g2/n-{}-k-{}-rep-{}.pkl".format(n,k,exps)
            comparator = Two_AgentsComparator(B = 100_000, K = k,n = n)
            scal1 = np.random.choice(scalar_list1, size = len(scalar_list1), replace = True)
            scal2 = np.random.choice(scalar_list2, size = len(scalar_list1), replace = True)
            comparator.compare_scalars(scal1, scal2)
            with open(filename, "wb") as f:
                pickle.dump(comparator.__dict__, f)
