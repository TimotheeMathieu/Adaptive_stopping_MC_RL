import numpy as np
from compare_agents import MultipleAgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('stat_precipice/stat_preci_data.pkl', 'rb') as handle:
    data = pickle.load(handle)


all_scalars = []
for k, v in data.items():
    all_scalars.append(v.flatten())
    sns.kdeplot(v.flatten(), label = k)


dict_res = {}
combs = [[10, 10], [20, 5], [25, 4], [50, 2]]
for comb in combs:
    for i in range(2):
        n, k = comb[i], comb[1-i]
        for exps in range(100):
            filename = "stat_precipice/n-{}-k-{}-rep-{}.pkl".format(n,k,exps)
            comparator = MultipleAgentsComparator(B = 100_000, K = k,n = n)
            rands_scal = []
            for scal_list in all_scalars:
                rands_scal.append(np.random.choice(scal_list, size = len(scal_list), replace = True))
            comparator.compare_scalars(rands_scal)
            with open(filename, "wb") as f:
                pickle.dump(comparator.__dict__, f)
# 
#
# plt.xlabel("evaluations means")
# plt.legend()
# plt.savefig("stat_precipice_algos.png")
