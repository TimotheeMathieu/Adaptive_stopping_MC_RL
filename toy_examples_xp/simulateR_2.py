import os
from utils import *
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import pandas as pd
from rlberry.utils.logging import set_level
set_level('ERROR')

import argparse
script_directory = os.path.realpath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('experiment', metavar='N', type=int, help='experiment identifier. Can be 1 or 2.')
parser.add_argument('--full-xp', dest='M', action='store_const',
                    const=5000, default=30,
                    help='Do the full xp (default use less iterations for faster computation)')

args = parser.parse_args()

def exp1(diff_means):
    mus = [0-diff_means/2, 0+diff_means/2]
    return make_different_agents(mus = mus)

def exp2(diff_means):
    mus = [0, diff_means]
    return make_different_agents(mus = mus)

def exp3(df):
    manager1 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": 0, "df": df, "type": "student"},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )
    manager2 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": 0, "std": 1.},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )
    return manager1, manager2


if __name__ == "__main__":
    
    # change the next two lines if you want to change the number of calls of AdaStop (M)
    M=args.M
    EXP = "exp"+str(args.experiment)

    SAVE = False
    alpha = 0.05

    res = []
    restime = []
    p_vals = []

    seeds = np.arange(M)
    K_list = np.array([5])
    n_list = np.array([5])
    B_list = np.array([10**4])
    if M < 100:
        dist_params_list = np.linspace(0, 1, 3) # reduced experiment
    else:
        dist_params_list = np.linspace(0, 1, 10)

    mesh = np.meshgrid(seeds, K_list, n_list, B_list, dist_params_list)

    seed_iter = mesh[0].reshape(-1)
    K_iter = mesh[1].reshape(-1)
    n_iter = mesh[2].reshape(-1)
    B_iter = mesh[3].reshape(-1)
    dist_params_iter = mesh[4].reshape(-1)

    num_comb = len(mesh[0].reshape(-1))


    def decision(save_results=SAVE, **kwargs):
        exp_name = kwargs["exp_name"]
        os.makedirs(os.path.join("mgres_alt", exp_name), exist_ok=True)
        filename = os.path.join("mgres_alt", exp_name, "result_K={}-n={}-B={}-dist_params={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["dist_params"], kwargs["seed"]))
        comparator = RlberryComparator(n = kwargs["n"], K= kwargs["K"],B= kwargs["B"], alpha= alpha, beta=0, seed=kwargs["seed"], n_evaluations=1)
        if exp_name == "exp1":
            manager1, manager2 = exp1(kwargs["dist_params"])
        elif exp_name == "exp2":
            manager1, manager2 = exp2(kwargs["dist_params"])
        elif exp_name == "exp3":
            manager1, manager2 = exp3(kwargs["dist_params"])
        else:
            raise ValueError
        comparator.compare([manager2, manager1])
        if save_results:
            with open(filename, "wb") as f:
                pickle.dump([kwargs, comparator], f)
        
        return comparator

    def decision_par(i):
        return decision(seed=seed_iter[i], n=n_iter[i],K= K_iter[i], B=B_iter[i],dist_params= dist_params_iter[i], exp_name = EXP)


    res = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(num_comb)))

    n_iters = {}
    decs = {}
    for i, comp in enumerate(res):
        dmu = dist_params_iter[i]

        if dmu in n_iters.keys():
            n_iters[dmu].append(list(comp.n_iters.values())[0] / 2)
        else:
            n_iters[dmu] = [list(comp.n_iters.values())[0] / 2]

        if dmu in decs.keys():
            decs[dmu].append(list(comp.decisions.values())[0])
        else:
            decs[dmu] = [list(comp.decisions.values())[0]]


    decs_df = pd.DataFrame(decs)
    n_iter_df = pd.DataFrame(n_iters)

    decs_df.to_csv(os.path.join(script_directory, "results", EXP +"_" + str(M) + "_decs.csv"))
    n_iter_df.to_csv(os.path.join(script_directory,"results",EXP+"_" +str(M) +"_niter.csv"))
    print("Done!")


