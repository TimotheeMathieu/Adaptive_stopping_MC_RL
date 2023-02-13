# import os
# os.chdir("/home/ashilova/Adaptive_stopping_MC_RL/code")

# print(os.getcwd())s


import sys
import os
sys.path.insert(0, "/home/rdellave/Adaptive_stopping_MC_RL/code")

# import os
# print(os.getcwd())
# os.chdir("/home/ashilova/Adaptive_stopping_MC_RL/code")
# print(os.getcwd())

from rlberry.agents import Agent
from rlberry.envs import Model
import rlberry.spaces as spaces
from compare_agents import Two_AgentsComparator
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import pickle



# GST definition

K = 1  # at most 4 groups
alpha = 0.05
n = 16  # size of a group

B = 10000

class RandomAgent(Agent):
    def __init__(self, env, drift=0, std = 1, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.drift = drift
        self.std = std

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        return self.drift + self.rng.normal(size=n_simulations)*self.std

#TODO check that comparator is using n_simulations
class MixtureGaussianAgent(Agent):
    def __init__(self, env, means=[0], stds=[1], prob_mixture = [1], **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.means = np.array(means)
        self.prob_mixture = np.array(prob_mixture)
        self.stds = np.array(stds)

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        idxs = self.rng.choice(np.arange(len(self.means)), size=n_simulations, p=self.prob_mixture)
        ret = self.means[idxs] + self.rng.normal(size=n_simulations)*self.stds[idxs]

        # print("Gaussians", idxs)
        # print("sampled_proba", sum(idxs)/ len(idxs))
        return ret



class DummyEnv(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        self.n_arms = 2
        self.action_space = spaces.Discrete(1)

    def step(self, action):
        pass

    def reset(self):
        return 0


def make_same_agents(diff_means, probas = [0.5, 0.5]):

    manager1 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0, 0+diff_means], "stds": [0.1, 0.1], "prob_mixture": probas},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )

    manager2 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0, 0 + diff_means], "stds": [0.1, 0.1], "prob_mixture": probas},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )
    return manager1, manager2

def exp1(diff_means):
    mus = [0-diff_means/2, 0+diff_means/2]
    return make_different_agents(mus = mus)

def exp2(diff_means):
    mus = [0, diff_means]
    return make_different_agents(mus = mus)

def make_different_agents(mus, probas = [0.5, 0.5]):

    manager1 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": mus, "stds": [0.1, 0.1], "prob_mixture": probas},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )

    manager2 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": 0, "std": 0.1},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )

    
    return manager1, manager2


if __name__ == "__main__":

    M=10000
    # max_num_seeds = 

    seeds = np.arange(M)
    # K_list = np.arange(4) + 1
    K_list = np.array([5])
    n_list = np.array([5])
    B_list = np.array([10**4])# 10**4, 5*10**4, 10**5])
    dmu_list = np.linspace(0, 1, 10)

    mesh = np.meshgrid(seeds, K_list, n_list, B_list, dmu_list)

    seed_iter = mesh[0].reshape(-1)
    K_iter = mesh[1].reshape(-1)
    n_iter = mesh[2].reshape(-1)
    B_iter = mesh[3].reshape(-1)
    dmu_iter = mesh[4].reshape(-1)

    num_comb = len(mesh[0].reshape(-1))


    #TODO finish this part
    # nk= np.unique(n_iter*K_iter)
    # mesh2 = np.meshgrid(seeds, nk, B_list, dmu_list)

    # seed_iter2 = mesh2[0].reshape(-1)
    # nk_iter = mesh[1].reshape(-1)
    # B_iter2 = mesh[2].reshape(-1)
    # dmu_iter2 = mesh[3].reshape(-1)

    # num_comb2 = len(mesh2[0].reshape(-1))
    #TODO






    res = []
    restime = []
    p_vals = []


    def decision(**kwargs):
        os.makedirs("mgres", exist_ok=True)
        filename = "mgres/result_K={}-n={}-B={}-dmu={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["diff_means"], kwargs["seed"])
        comparator = Two_AgentsComparator(kwargs["n"], kwargs["K"], kwargs["B"],  alpha, seed=kwargs["seed"])
        # manager1, manager2 = make_same_agents(kwargs["diff_means"])
        manager1, manager2 = exp2(kwargs["diff_means"])
        comparator.compare(manager2, manager1)
        with open(filename, "wb") as f:
            pickle.dump([kwargs, comparator.__dict__], f)
        
        return comparator.decision, comparator.n_iter / 2

    def decision_par(i):
        return decision(seed=seed_iter[i], n=n_iter[i],K= K_iter[i], B=B_iter[i],diff_means= dmu_iter[i])

    #TODO
    # def non_adaptive_decision(**kwargs):
    #     K = 1
    #     os.makedirs("mgres", exist_ok=True)
    #     filename = "mgres/result_nonada_K={}-n={}-B={}-dmu={}-seed={}.pickle".format(K, kwargs["nk"], kwargs["B"], kwargs["diff_means"], kwargs["seed"])
    #     comparator = Two_AgentsComparator(kwargs["nk"], K, kwargs["B"],  alpha, seed=kwargs["seed"])
    #     manager1, manager2 = make_same_agents(kwargs["diff_means"])
    #     comparator.compare(manager2, manager1)
    #     with open(filename, "wb") as f:
    #         pickle.dump(kwargs, f)
    #         pickle.dump(comparator.__dict__, f)
        
    #     return comparator.decision, comparator.n_iter / 2

    # def non_adaptive_decision_par(i):
    #     return non_adaptive_decision(seed=seed_iter2[i], nk=nk_iter[i], B=B_iter2[i], diff_means= dmu_iter2[i])
    #TODO

    res = Parallel(n_jobs=24, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(num_comb)))
    print("Done!")
    # res2 = Parallel(n_jobs=6, backend="multiprocessing")(delayed(non_adaptive_decision_par)(i) for i in tqdm(range(num_comb2)))
    # decision(0)
    # idxs = np.array([i[0] for i in res]) == "accept"
    # # idxs2 = np.array([i[0] for i in res2]) == "accept"


    # #print("mean running time", np.mean(np.array(restime)[idxs]))
    # print("proba to reject in adaptive", np.mean(1 - idxs))
    # # print("proba to reject in non adaptive", np.mean(1 - idxs2))
    # print("proba to accept", np.mean(idxs))
