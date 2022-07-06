from rlberry.agents.torch import A2CAgent
from rlberry.agents import Agent
from rlberry.envs import gym_make, Model
import rlberry.spaces as spaces
from compare_agents import AgentComparator
import time
import numpy as np
# GST definition

K = 3  # at most 5 groups
alpha = 0.05
n = 5 # size of a group

comparator = AgentComparator(n, K, alpha)

# DeepRL agent definition
env_ctor = gym_make
env_kwargs = dict(id="CartPole-v1")
seed = 42
budget = 1e2


class RandomAgent(Agent):
    def __init__(self, env, drift=0, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.drift = drift
    def fit(self, budget: int, **kwargs):
        pass
    def eval(self, n_simulations=1, **kwargs):
        return self.drift + self.rng.normal(size=n_simulations)
    
class DummyEnv(Model):
    def __init__(self,  **kwargs):
        Model.__init__(self, **kwargs)
        self.n_arms = 2
        self.action_space = spaces.Discrete(1)

    def step(self, action):
        pass

    def reset(self):
        return 0


np.random.seed(seed)
if __name__ == "__main__":
    
    # manager1 = (
    #     A2CAgent,
    #     dict(train_env=(env_ctor, env_kwargs),
    #     fit_budget=budget,
    #     seed=seed,
    #     eval_kwargs=dict(eval_horizon=500),
    #     init_kwargs=dict(
    #         learning_rate=1e-3, entr_coef=0.0 
    #     ),
    #     parallelization="process",
    #     mp_context="forkserver",)
    # )
    # manager2 = (
    #     A2CAgent,
    #     dict(train_env=(env_ctor, env_kwargs),
    #     fit_budget=budget,
    #     seed=seed,
    #     init_kwargs=dict(
    #         learning_rate=1e-3,  
    #         entr_coef=0.0,
    #         #batch_size=1024,
    #     ),
    #     eval_kwargs=dict(eval_horizon=500),
    #     agent_name="A2C_tuned",
    #     parallelization="process",
    #     mp_context="forkserver",)
    # )


    manager1 =(
        RandomAgent,
        dict(train_env=(DummyEnv, {}),
        init_kwargs = {"drift":0},
        fit_budget=1,
        agent_name = "Agent1"))
    
    manager2 = (
        RandomAgent,
        dict(train_env=(DummyEnv, {}),
        init_kwargs = {"drift":0},
        fit_budget=1,
        agent_name = "Agent2"))
    
    M = 1
    res = []
    for _ in range(M):
        a = time.time()
        comparator.compare(manager2, manager1)
        res.append(comparator.decision)
        print("Time: ",time.time()-a)
    
