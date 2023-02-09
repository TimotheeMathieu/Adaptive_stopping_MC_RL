import numpy as np
from compare_agents import Two_AgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
ks = []
scalar_list1 = np.loadtxt("evals_sac_hc.txt")
scalar_list2 = np.loadtxt("evals_td3_hc.txt")
sns.set_style("whitegrid")
sns.kdeplot(scalar_list1, x = "evaluation means", label = "SAC")
sns.kdeplot(scalar_list2, x = "evaluation means", label  = "TD3")
plt.xlabel("evaluations means")
plt.legend()
plt.savefig("sac_vs_td3_hc.png")
all_comparator_stuff = []
res = np.zeros((5, 5))
for n in range(1,6):
    for k in range(1,6):
        avg = []
        for exps in range(100):
            comparator = Two_AgentsComparator(B = 100_000, K = k,n = n)

            scal1 = np.random.choice(scalar_list1, size = len(scalar_list1), replace = True)
            scal2 = np.random.choice(scalar_list2, size = len(scalar_list1), replace = True)

            stuff_comparator = comparator.compare_scalars(scal1, scal2)
            all_comparator_stuff.append(stuff_comparator)
            avg.append(comparator.n_iter / 2)
        res[n-1,k-1]=np.array(avg).mean()

np.save("table_res.npy", res )
np.save("results_comparator.npy", all_comparator_stuff)
