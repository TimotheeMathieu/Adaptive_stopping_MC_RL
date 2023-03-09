import numpy as np
from compare_agents import MultipleAgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json

scalar_list1 = np.loadtxt("h2g2/evals_sac_hc.txt")
scalar_list2 = np.loadtxt("h2g2/evals_td3_hc.txt")
dist_means = np.abs(np.mean(scalar_list1)-np.mean(scalar_list2))
pool_var = np.sqrt((np.std(scalar_list1)**2+np.std(scalar_list2)**2)/2)
print("relative effect size = ", dist_means/pool_var)
parameters = {}
parameters["nb_exps"] = int(1e3)
combs = [[3, 7], [4, 5], [4, 4], [3, 6], [3,8], [5, 5]]
# combs = [[3, 7]]

parameters["nk"] = combs
parameters["B"] = 10_000
save_path = "code/h2g2/"
parameters["save_path"] = save_path
parameters["numpy_version"] = np.__version__
#save parameters
with open("h2g2/parameters.json", "w") as file:
    file.write(json.dumps(parameters))


dic_results = {}
for comb in parameters["nk"]:
    for i in range(2):
        n, k = comb[i], comb[1-i]
        effective_comparisons, true_positives = 0 , 0
        for exps in range(parameters["nb_exps"]):
            # filename = "h2g2/n-{}-k-{}-rep-{}.pkl".format(n,k,exps)
            comparator = MultipleAgentsComparator(B = 10_000, K = k,n = n)
            scal1 = np.random.choice(scalar_list1, size = len(scalar_list1), replace = True)
            scal2 = np.random.choice(scalar_list2, size = len(scalar_list1), replace = True)
            comparator.compare_scalars([scal1, scal2])
            rejects = ["reject" == deci for deci in comparator.decisions]
            true_positives += len(rejects) == 1 and rejects[0]
            # print(comparator.n_iters[0])
            effective_comparisons += comparator.n_iters[0]

        dic_results["n={}, k={}".format(n,k)] = {"true_positives": true_positives/parameters["nb_exps"], "effective_comparisons": effective_comparisons/parameters["nb_exps"]}
with open("h2g2/res_true_positives.json", "w") as f:
    f.write(json.dumps(dic_results))
