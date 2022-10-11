from rlberry.agents import Agent
from rlberry.envs import Model
import rlberry.spaces as spaces
import sys
sys.path.append('../')
from compare_agents import Two_AgentsComparator
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed



# GST definition
alpha = 0.05
K = 5
mup = 500
muq = 470
sigmap = 40
sigmaq = 40
n = 5

B = 10000

class RandomAgent(Agent):
    def __init__(self, env, drift=0, sigma = 1, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.drift = drift
        self.sigma = sigma

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        return self.drift + self.sigma * self.rng.normal(size=n_simulations)

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

    manager1 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": mup, "sigma" : sigmap},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )

    manager2 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": muq, "sigma" : sigmaq},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )

    res = []

    def decision(seed):
        comparator = Two_AgentsComparator(n, K,B,  alpha, seed=seed)
        comparator.compare(manager2, manager1)
        return comparator.decision
    res = Parallel(n_jobs=6, backend="multiprocessing")(delayed(decision)(i) for i in tqdm(range(2000)))
    idxs = np.array(res) == "accept"
    #print("mean running time", np.mean(np.array(restime)[idxs]))
    print("proba to reject", np.mean(1 - idxs))
