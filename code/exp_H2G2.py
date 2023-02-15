import numpy as np
from compare_agents import MultipleAgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json

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

parameters = {}
parameters["nb_exps"] = 100
combs = [[3, 7], [4, 5], [4, 4], [2, 10], [3, 6]]
parameters["nk"] = combs
parameters["B"] = 100_000
save_path = "code/h2g2/"
parameters["save_path"] = save_path
parameters["numpy_version"] = np.__version__
#save parameters
with open("h2g2/parameters.json", "w") as file:
    file.write(json.dumps(parameters))


for comb in combs:
    for i in range(2):
        n, k = comb[i], comb[1-i]
        for exps in range(100):
            filename = "h2g2/n-{}-k-{}-rep-{}.pkl".format(n,k,exps)
            comparator = MultipleAgentsComparator(B = 100_000, K = k,n = n)
            scal1 = np.random.choice(scalar_list1, size = len(scalar_list1), replace = True)
            scal2 = np.random.choice(scalar_list2, size = len(scalar_list1), replace = True)
            comparator.compare_scalars([scal1, scal2])
            with open(filename, "wb") as f:
                pickle.dump(comparator.__dict__, f)
