# import sys
import os
# sys.path.insert(0, "/home/rdellave/Adaptive_stopping_MC_RL/adastop")
# sys.path.insert(0, "/home/ashilova/Adaptive_stopping_MC_RL/adastop")
# print(sys.path)
# from rlberry.agents import Agent
# from rlberry.envs import Model
# import rlberry.spaces as spaces
from adastop import MultipleAgentsComparator
from compare_agents import MultipleAgentsComparator
from utils import *
# import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

# GST definition
K = 16  # at most 4 groups
alpha = 0.05
n = 1  # size of a group
B = 10000


# class DummyEnv(Model):
#     def __init__(self, **kwargs):
#         Model.__init__(self, **kwargs)
#         self.n_arms = 2
#         self.action_space = spaces.Discrete(1)

#     def step(self, action):
#         pass

#     def reset(self):
#         return 0

# class RandomAgent(Agent):
#     def __init__(self, env, drift=0, std = 1, **kwargs):
#         Agent.__init__(self, env)
#         self.drift = drift
#         self.std = std
#         if "type" in kwargs.keys():
#             self.type = kwargs["type"]
#         else:
#             self.type = "normal"
#         self.kwargs = kwargs

#     def fit(self, budget: int, **kwargs):
#         pass

#     def eval(self, n_simulations=1, **kwargs):
#         if self.type == "normal":
#             noise = self.rng.normal(size=n_simulations)*self.std
#         elif self.type == "student":
#             if "df" in self.kwargs.keys():
#                 df = self.kwargs["df"]
#             else:
#                 df = 2.
#             noise = self.rng.standard_t(df, size=n_simulations)
#         return self.drift + noise


# #TODO check that comparator is using n_simulations
# class MixtureGaussianAgent(Agent):
#     def __init__(self, env, means=[0], stds=[1], prob_mixture = [1], **kwargs):
#         Agent.__init__(self, env, **kwargs)
#         self.means = np.array(means)
#         self.prob_mixture = np.array(prob_mixture)
#         self.stds = np.array(stds)

#     def fit(self, budget: int, **kwargs):
#         pass

#     def eval(self, n_simulations=1, **kwargs):
#         idxs = self.rng.choice(np.arange(len(self.means)), size=n_simulations, p=self.prob_mixture)
#         ret = self.means[idxs] + self.rng.normal(size=n_simulations)*self.stds[idxs]
#         return ret


# def make_same_agents(diff_means, probas = [0.5, 0.5]):
#     manager1 = (
#         MixtureGaussianAgent,
#         dict(
#             train_env=(DummyEnv, {}),
#             init_kwargs={"means": [0, 0+diff_means], "stds": [0.1, 0.1], "prob_mixture": probas},
#             fit_budget=1,
#             agent_name="Agent1",
#         ),
#     )
#     manager2 = (
#         MixtureGaussianAgent,
#         dict(
#             train_env=(DummyEnv, {}),
#             init_kwargs={"means": [0, 0 + diff_means], "stds": [0.1, 0.1], "prob_mixture": probas},
#             fit_budget=1,
#             agent_name="Agent2",
#         ),
#     )
#     return manager1, manager2


# def make_different_agents(mus, probas = [0.5, 0.5]):
#     manager1 = (
#         MixtureGaussianAgent,
#         dict(
#             train_env=(DummyEnv, {}),
#             init_kwargs={"means": mus, "stds": [0.1, 0.1], "prob_mixture": probas},
#             fit_budget=1,
#             agent_name="Agent1",
#         ),
#     )
#     manager2 = (
#         RandomAgent,
#         dict(
#             train_env=(DummyEnv, {}),
#             init_kwargs={"drift": 0, "std": 0.1},
#             fit_budget=1,
#             agent_name="Agent2",
#         ),
#     )
#     return manager1, manager2

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



    res = []
    restime = []
    p_vals = []

    M=5000
    EXP = "exp1"
    # max_num_seeds = 

    seeds = np.arange(M)
    # K_list = np.arange(4) + 1
    K_list = np.array([5])
    n_list = np.array([5])
    B_list = np.array([10**4])# 10**4, 5*10**4, 10**5])
    dist_params_list = np.linspace(0, 1, 10)
    # dist_params_list = np.array([2., 4., 8., 64., 1024.])

    mesh = np.meshgrid(seeds, K_list, n_list, B_list, dist_params_list)

    seed_iter = mesh[0].reshape(-1)
    K_iter = mesh[1].reshape(-1)
    n_iter = mesh[2].reshape(-1)
    B_iter = mesh[3].reshape(-1)
    # dmu_iter = mesh[4].reshape(-1)
    dist_params_iter = mesh[4].reshape(-1)

    num_comb = len(mesh[0].reshape(-1))




    def decision(**kwargs):
        exp_name = kwargs["exp_name"]
        os.makedirs(os.path.join("mgres_alt", exp_name), exist_ok=True)
        filename = os.path.join("mgres_alt", exp_name, "result_K={}-n={}-B={}-dist_params={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["dist_params"], kwargs["seed"]))
        comparator = MultipleAgentsComparator(n = kwargs["n"], K= kwargs["K"],B= kwargs["B"], alpha= alpha, beta=0, seed=kwargs["seed"])
        # manager1, manager2 = make_same_agents(kwargs["diff_means"])
        if exp_name == "exp1":
            manager1, manager2 = exp1(kwargs["dist_params"])
        elif exp_name == "exp2":
            manager1, manager2 = exp2(kwargs["dist_params"])
        elif exp_name == "exp3":
            manager1, manager2 = exp3(kwargs["dist_params"])
        else:
            raise ValueError
        comparator.compare([manager2, manager1])
        with open(filename, "wb") as f:
            pickle.dump([kwargs, comparator], f)
        
        return comparator

    def decision_par(i):
        return decision(seed=seed_iter[i], n=n_iter[i],K= K_iter[i], B=B_iter[i],dist_params= dist_params_iter[i], exp_name = EXP)



    res = Parallel(n_jobs=-5, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(num_comb)))

    print("Done!")


