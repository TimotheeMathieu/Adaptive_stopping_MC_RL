import sys
sys.path.insert(0, "/home/rdellave/Adaptive_stopping_MC_RL/code")
print(sys.path)


# import sys
# sys.path.insert(0, "/home/ashilova/Adaptive_stopping_MC_RL/code")


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

K = 16  # at most 4 groups
alpha = 0.05
n = 1  # size of a group

B = 10000

class RandomAgent(Agent):
    def __init__(self, env, drift=0, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.drift = drift

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        return self.drift + self.rng.normal(size=n_simulations)

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


def make_same_agents(diff_means):

    manager1 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0, 0+diff_means], "stds": [0.1, 0.1], "prob_mixture": [0.5, 0.5]},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )

    manager2 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0, 0 + diff_means], "stds": [0.1, 0.1], "prob_mixture": [0.5, 0.5]},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )
    return manager1, manager2



if __name__ == "__main__":

    M=5
    seeds = np.arange(M)
    K_list = np.arange(2)+1
    n_list = np.arange(1)*2 + 1
    B_list = np.array([5*10**3,])# 10**4, 5*10**4, 10**5])
    dmu_list = np.linspace(0, 1, 1)

    mesh = np.meshgrid(seeds, K_list, n_list, B_list, dmu_list)

    seed_iter = mesh[0].reshape(-1)
    K_iter = mesh[1].reshape(-1)
    n_iter = mesh[2].reshape(-1)
    B_iter = mesh[3].reshape(-1)
    dmu_iter = mesh[4].reshape(-1)

    num_comb = len(mesh[0].reshape(-1))






    # manager1 = (
    #     MixtureGaussianAgent,
    #     dict(
    #         train_env=(DummyEnv, {}),
    #         init_kwargs={"means": [0,2], "stds": [0.1, 0.1], "prob_mixture": [0.5, 0.5]},
    #         fit_budget=1,
    #         agent_name="Agent1",
    #     ),
    # )

    # manager2 = (
    #     MixtureGaussianAgent,
    #     dict(
    #         train_env=(DummyEnv, {}),
    #         init_kwargs={"means": [0,2], "stds": [0.1, 0.1], "prob_mixture": [0.5, 0.5]},
    #         fit_budget=1,
    #         agent_name="Agent2",
    #     ),
    # )

    # manager1 = (
    #     RandomAgent,
    #     dict(
    #         train_env=(DummyEnv, {}),
    #         init_kwargs={"drift": 0},
    #         fit_budget=1,
    #         agent_name="Agent1",
    #     ),
    # )


    # manager2 = (
    #     RandomAgent,
    #     dict(
    #         train_env=(DummyEnv, {}),
    #         init_kwargs={"drift": 0},
    #         fit_budget=1,
    #         agent_name="Agent2",
    #     ),
    # )

    M = 70
    res = []
    restime = []
    p_vals = []


    def decision(**kwargs):
        os.makedirs("mgres", exist_ok=True)
        filename = "mgres/result_K={}-n={}-B={}-dmu={}-seed={}.pickle".format(kwargs["K"], kwargs["n"], kwargs["B"], kwargs["diff_means"], kwargs["seed"])
        comparator = Two_AgentsComparator(kwargs["n"], kwargs["K"], kwargs["B"],  alpha, seed=kwargs["seed"])
        manager1, manager2 = make_same_agents(kwargs["diff_means"])
        comparator.compare(manager2, manager1)
        with open(filename, "wb") as f:
            pickle.dump(kwargs, f)
            pickle.dump(comparator.__dict__, f)
        
        return comparator.decision, comparator.n_iter / 2

    def decision_par(i):
        return decision(seed=seed_iter[i], n=n_iter[i],K= K_iter[i], B=B_iter[i],diff_means= dmu_iter[i])

    # decision_par = lambda i: decision(seed_iter[i], n_iter[i], K_iter[i], B_iter[i], dmu_iter[i])

    print("checkpoint")

    res = Parallel(n_jobs=6, backend="multiprocessing")(delayed(decision_par)(i) for i in tqdm(range(num_comb)))
    # decision(0)
    idxs = np.array(res) == "accept"

    #print("mean running time", np.mean(np.array(restime)[idxs]))
    print("proba to reject", np.mean(1 - idxs))
    print("proba to accept", np.mean(idxs))
