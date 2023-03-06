from rlberry.agents import Agent
from rlberry.envs import Model
import rlberry.spaces as spaces
from compare_agents import Two_AgentsComparator
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


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
            init_kwargs={"drift": 0},
            fit_budget=1,
            agent_name="Agent1",
        ),
    )

    manager2 = (
        RandomAgent,
        dict(
            train_env=(DummyEnv, {}),
            init_kwargs={"drift": 0},
            fit_budget=1,
            agent_name="Agent2",
        ),
    )

    M = 200
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
    res = Parallel(n_jobs=6, backend="multiprocessing")(delayed(decision)(i) for i in tqdm(range(500)))
    idxs = np.array(res) == "accept"
    #print("mean running time", np.mean(np.array(restime)[idxs]))
    print("proba to reject", np.mean(1 - idxs))