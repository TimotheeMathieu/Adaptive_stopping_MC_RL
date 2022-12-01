from rlberry.agents import Agent
from rlberry.envs import Model
import rlberry.spaces as spaces

import sys
sys.path.insert(0, "../")

from compare_agents import MultipleAgentsComparator
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from rlberry.utils.logging import set_level
set_level('ERROR')

# GST definition

K = 4  # at most 4 groups
alpha = 0.05
n = 4  # size of a group

B = 10000

class RandomAgent(Agent):
    def __init__(self, env, drift=0, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.drift = drift

    def fit(self, budget: int, **kwargs):
        pass

    def eval(self, n_simulations=1, **kwargs):
        return self.drift + self.rng.normal(size=n_simulations)


class DummyEnv(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        self.action_space = spaces.Discrete(1)

    def step(self, action):
        pass

    def reset(self):
        return 0


if __name__ == "__main__":

    managers = [(
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": 0},
            fit_budget=1,
            agent_name="Agent"+str(k),
        ),
    ) for k in range(4)]



    M = 200
    res = []
    restime = []
    p_vals = []

    def decision(seed):
        comparator = MultipleAgentsComparator(n, K,B,  alpha, seed=seed, beta = 0.5, joblib_backend = "multiprocessing")
        comparator.compare(managers)
        print(comparator.n_iters)
        return comparator.decisions
    #res = Parallel(n_jobs=6, backend="multiprocessing")(delayed(decision)(i) for i in tqdm(range(500)))
    res = [decision(i) for i in tqdm(range(200))]

    idxs = [ 'reject' in a for a in res]
    print("proba to reject", np.mean(idxs))
