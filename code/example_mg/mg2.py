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


if __name__ == "__main__":

    # mga = MixtureGaussianAgent(DummyEnv(), means=[0, 2], stds=[0.1, 1], prob_mixture=[0.3, 0.7])
    # print(mga.eval(100))

    manager1 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0,2], "stds": [0.1, 1], "prob_mixture": [0.3, 0.7]},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )

    manager2 = (
        MixtureGaussianAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"means": [0,2], "stds": [0.1, 1], "prob_mixture": [0.3, 0.7]},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )

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


    def decision(seed):
        comparator = Two_AgentsComparator(n, K,B,  alpha, seed=seed)
        comparator.compare(manager2, manager1)
        return comparator.decision
    # for _ in tqdm(range(M)):
    #     a = time.time()
    #     comparator = Two_AgentsComparator(n, K,B,  alpha)
    #     comparator.compare(manager2, manager1)
    #     res.append(comparator.decision)
    #     p_vals.append(comparator.p_val)
    #     restime.append(time.time()-a)
    res = Parallel(n_jobs=6, backend="multiprocessing")(delayed(decision)(i) for i in tqdm(range(M)))
    idxs = np.array(res) == "accept"

    #print("mean running time", np.mean(np.array(restime)[idxs]))
    print("proba to reject", np.mean(1 - idxs))
    print("proba to accept", np.mean(idxs))
