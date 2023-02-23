from rlberry.agents import Agent
from rlberry.envs import Model
import rlberry.spaces as spaces
import sys
sys.path.insert(0, "/home/rdellave/Adaptive_stopping_MC_RL/code")
# print(sys.path)
from compare_agents import MultipleAgentsComparator
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import pickle

from rlberry.utils.logging import set_level
set_level('ERROR')

class DummyEnv(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        self.n_arms = 2
        self.action_space = spaces.Discrete(1)

    def step(self, action):
        pass

    def reset(self):
        return 0

class RandomAgent(Agent):
    def __init__(self, env, drift=0, std = 1, **kwargs):
        Agent.__init__(self, env)
        self.drift = drift
        self.std = std
        if "type" in kwargs.keys():
            self.type = kwargs["type"]
        else:
            self.type = "normal"
        self.kwargs = kwargs

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations = 1, **kwargs):
        if self.type == "normal":
            noise = self.rng.normal(scale =self.std, size=n_simulations)
        elif self.type == "student":
            if "df" in self.kwargs.keys():
                df = self.kwargs["df"]
            else:
                df = 2.
            noise = self.rng.standard_t(df, size=n_simulations)
        return self.drift + noise


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
        return ret


def create_agents(agent_name, **kwargs):
    if agent_name == "mixture":
        assert "mus" in kwargs.keys() and "probas" in kwargs.keys()
        manager =  (
                        MixtureGaussianAgent,
                        dict(
                            train_env=(DummyEnv, {}),
                            init_kwargs={"means": kwargs["mus"], "stds": [0.1, 0.1], "prob_mixture": kwargs["probas"]},
                            fit_budget=1,
                            agent_name="Agent1",
                        ),
                    )
    elif agent_name == "single":
        init_kwargs = dict(type = kwargs["type"], drift = kwargs["drift"])
        if kwargs["type"] == "student":
            init_kwargs["df"] = kwargs["df"]
            manager =  (
                            RandomAgent,
                            dict(
                                train_env=(DummyEnv, {}),
                                init_kwargs=init_kwargs,
                                fit_budget=1,
                                agent_name="Agent1",
                            ),
                        )   
        elif kwargs["type"] == "normal":
            manager =  (
                            RandomAgent,
                            dict(
                                train_env=(DummyEnv, {}),
                                fit_budget=1,
                                agent_name="Agent1",
                            ),
                        )
        else: 
            raise ValueError

    return manager


def exp4():
    # managers = [create_agents("mixture", mus=[1.,-1.], probas=[0.5,0.5]),create_agents("mixture", mus=[1.,-1.], probas=[0.5,0.5]),
    #             create_agents("mixture", mus=[1.,-1.], probas=[0.5,0.5]),create_agents("mixture", mus=[1.,-1.], probas=[0.5,0.5]),
    #             create_agents("mixture", mus=[1.,-1.], probas=[0.5,0.5])]
    managers = [create_agents("single", type = "normal", drift =0),create_agents("single", type = "normal", drift =0),
                create_agents("single", type = "normal", drift =0),create_agents("single", type = "normal", drift =0),
                create_agents("single", type = "normal", drift =0)]
    return managers


if __name__ == "__main__":

    # managers = [(
    #     RandomAgent,
    #     dict(
    #         train_env=(DummyEnv, {}),
    #         init_kwargs={"drift": 0},
    #         fit_budget=1,
    #         agent_name="Agent"+str(k),
    #     ),
    # ) for k in range(3)]

    EXP = "exp4"
    K = 5  # at most 4 groups
    alpha = 0.05
    n = 5  # size of a group
    B = 10000
    M = 100

    res = []
    restime = []
    p_vals = []

    seeds = np.arange(M)
    # K_list = np.arange(4) + 1
    K_list = np.array([5])
    n_list = np.array([5])
    B_list = np.array([10**4])# 10**4, 5*10**4, 10**5])
    # dmu_list = np.linspace(0, 1, 10)
    dist_params_list = np.array([2., 4., 8., 64., 1024.])

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
        os.makedirs(os.path.join("different_distribution_res", exp_name), exist_ok=True)
        filename = os.path.join("different_distribution_res", exp_name, "result_K={}-n={}-B={}-dist_params={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["dist_params"], kwargs["seed"]))
        comparator = MultipleAgentsComparator(n = kwargs["n"], K = kwargs["K"], B = kwargs["B"], alpha = alpha, seed=kwargs["seed"], beta=0.01)
        
        if exp_name == "exp4":
            managers = exp4()
        else:
            raise ValueError
        
        comparator.compare(managers, verbose = False)

        with open(filename, "wb") as f:
            pickle.dump([kwargs, comparator.__dict__], f)
        
        return comparator.decisions

    def decision_par(i):
        return decision(seed = seed_iter[i], n = n_iter[i], K = K_iter[i], B = B_iter[i], dist_params = dist_params_iter[i], exp_name = EXP)

    # def decision(seed):
    #     comparator = MultipleAgentsComparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 10, beta=0.01)
    #     # comparator  = Comparator(n=n, K=K, B=B, alpha=alpha, beta=0.5, joblib_backend = "multiprocessing")
    #     comparator.compare(managers)
    #     print(comparator.n_iters)
    #     return comparator.decisions    


    res = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(M))) # n_jobs=1 for debugging
    # res = [decision_par(i) for i in tqdm(range(M))]


    # estimate of the Family-Wise Error Rate (FWER)
    idxs = [ 'reject' in a for a in res]
    print(res)
    print("proba to reject", np.mean(idxs))


    # print(comparator.decisions,
    #   comparator.comparisons,
    #   comparator.rejected_decision)