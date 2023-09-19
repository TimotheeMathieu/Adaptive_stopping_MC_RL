
import numpy as np
from adastop import MultipleAgentsComparator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json
from tqdm import tqdm
import argparse
import os
import sys

script_directory = os.path.realpath(os.path.dirname(__file__))



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--full-xp', dest='nb_runs', action='store_const',
                    const=int(1e3), default=50,
                    help='Do the full xp (default use less iterations for faster computation)')

args = parser.parse_args()

scalar_list1 = np.loadtxt(os.path.join(script_directory, "data/evals_sac_hc.txt"))
scalar_list2 = np.loadtxt(os.path.join(script_directory,"data/evals_td3_hc.txt"))

dist_means = np.abs(np.mean(scalar_list1)-np.mean(scalar_list2))
pool_var = np.sqrt((np.std(scalar_list1)**2+np.std(scalar_list2)**2)/2)
print("relative effect size = ", dist_means/pool_var)

seed = 42
np.random.seed(seed)

parameters = {}
parameters["seed"] = seed
parameters["nb_exps"] = args.nb_runs
combs = []
for n in range(2, 7): 
    for k in range(1, 6): # change to range(1,9) to recover the results of the article.
        combs.append([n,k])

parameters["nk"] = combs
parameters["B"] = 10_000
save_path = "code/h2g2/"
parameters["save_path"] = save_path
parameters["numpy_version"] = np.__version__

#save parameters
with open(os.path.join(script_directory,"data/parameters.json"), "w") as file:
    file.write(json.dumps(parameters))

dic_results = {}
for comb in tqdm(parameters["nk"]):
    n, k = comb[0], comb[1]
    effective_comparisons, true_positives = [] , []
    for exps in range(parameters["nb_exps"]):
        comparator = MultipleAgentsComparator(B = parameters["B"], K = k,n = n)
        scal1 = np.random.choice(scalar_list1, size = len(scalar_list1), replace = True)
        scal2 = np.random.choice(scalar_list2, size = len(scalar_list1), replace = True)
        
        comparator.partial_compare({"SAC":scal1,"TD3": scal2}, verbose=False)
        rejects = ("smaller" in comparator.decisions.values()) or ("larger" in comparator.decisions.values())
        effective_comparisons.append(comparator.n_iters['SAC']) # only two agents. The number of iteration are the same for the two agents.

        true_positives.append(rejects)
    effective_comparisons_array = np.array(effective_comparisons)
    true_positives_array = np.array(true_positives)

    dic_results["n={}, k={}".format(n,k)] = {"true_positives": true_positives_array.sum()/parameters["nb_exps"], "effective_comparisons": effective_comparisons_array.sum()/parameters["nb_exps"]}

print(dic_results)
with open("results/results_non_adaptive_xp.json", "w") as f:
    f.write(json.dumps(dic_results))
