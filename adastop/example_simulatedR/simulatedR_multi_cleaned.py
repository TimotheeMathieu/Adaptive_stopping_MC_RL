import os
home_folder = os.environ["HOME"]
workdir = os.path.join(home_folder,"Adaptive_stopping_MC_RL","adastop")
from adastop import MultipleAgentsComparator
from utils import *
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle


from rlberry.utils.logging import set_level
set_level('ERROR')


labels = ["N", "*N", "*MG1", "MG1", "MG2", "tS1", "MG3", "*MG3", "MtS", "tS2"]

def exp4():
    managers = [create_agents("single", labels[0], type = "normal", drift =0),
                create_agents("single", labels[1], type = "normal", drift =0),
                create_agents("mixture", labels[2], mus = [1.,-1.], probas=[0.5,0.5]),
                create_agents("mixture", labels[3], mus = [1.,-1.], probas=[0.5,0.5]),
                create_agents("mixture", labels[4], mus = [0.2,-0.2], probas=[0.5,0.5]),
                create_agents("single", labels[5], type = "student", df = 3, drift = 0.),#
                create_agents("mixture", labels[6], mus = [0.,8.], probas=[0.5,0.5]),
                create_agents("mixture", labels[7], mus = [0.,8.], probas=[0.5,0.5]),
                create_agents("mixture", labels[8], type = "student", mus = [0. ,8.], probas = [0.5,0.5], df = 3),#
                create_agents("single", labels[9], type = "student", drift = 8., df=3),
                ]
    
    return managers

#The list of equal comparisons
true_hyp = [[0,1], [2,3], [6,7]]
true_hyp_means = [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4], [0,5], [1,5], [2,5], [3,5], [4,5], [6,7], [6,8], [7,8]]


if __name__ == "__main__":

    res = []
    restime = []
    p_vals = []

    EXP = "exp4"
    alpha = 0.05
    M = 100

    seeds = np.arange(M)
    K_list = np.array([5])
    n_list = np.array([5])
    B_list = np.array([10**4])
 

    mesh = np.meshgrid(seeds, K_list, n_list, B_list)

    seed_iter = mesh[0].reshape(-1)
    K_iter = mesh[1].reshape(-1)
    n_iter = mesh[2].reshape(-1)
    B_iter = mesh[3].reshape(-1)

    num_comb = len(mesh[0].reshape(-1))

    def decision(**kwargs):
        exp_name = kwargs["exp_name"]
        os.makedirs(os.path.join(workdir, "example_simulatedR", "multc_sd_res", exp_name), exist_ok=True)
        filename = os.path.join(workdir, "example_simulatedR", "multc_sd_res", exp_name, "result_K={}-n={}-B={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["seed"]))
        comparator = MultipleAgentsComparator(n = kwargs["n"], K = kwargs["K"], B = kwargs["B"], alpha = alpha, seed=kwargs["seed"], beta=0.01, n_evaluations=1)
        if exp_name == "exp4":
            managers = exp4()
        else:
            raise ValueError
        comparator.compare(managers, verbose = False)
        with open(filename, "wb") as f:
            pickle.dump([kwargs, comparator], f)
        return comparator

    def decision_par(i):
        return decision(seed = seed_iter[i], n = n_iter[i], K = K_iter[i], B = B_iter[i], exp_name = EXP)

    res = Parallel(n_jobs=-5, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(num_comb))) 
    nrej = 0
    nrej_means= 0
    n_tests = len(res)
    for comp in res:
        for h in true_hyp:
            key = "[{} {}]".format(h[0], h[1])
            if comp.decisions[key] != "equal":
                nrej += 1
                break


        for h in true_hyp_means:
            key = "[{} {}]".format(h[0], h[1])
            if comp.decisions[key] != "equal":
                nrej_means += 1
                break


    FWE = nrej/n_tests

    FWE_means = nrej_means/ n_tests

    print("FWE = {}, FWE for means={}".format(FWE, FWE_means))



