from rlberry.agents.torch import A2CAgent, PPOAgent
from rlberry.envs import gym_make
from compare_agents import AgentComparator
import numpy as np


# GST definition

K = 5  # at most 5 groups
alpha = 0.05
n = 4  # size of a group. i.e. 4 fits at a time.

comparator = AgentComparator(n, K, alpha, n_evaluations=10)

# DeepRL agent definition
env_ctor = gym_make
env_kwargs = dict(id="CartPole-v1")
seed = 42
budget = 1e4


np.random.seed(seed)
if __name__ == "__main__":

    manager1 = (
        A2CAgent,
        dict(
            train_env=(env_ctor, env_kwargs),
            fit_budget=budget,
            eval_kwargs=dict(eval_horizon=500),
            init_kwargs=dict(learning_rate=1e-3, entr_coef=0.0),
            parallelization="process",
            agent_name="A2C",
        ),
    )
    manager2 = (
        PPOAgent,
        dict(
            train_env=(env_ctor, env_kwargs),
            fit_budget=budget,
            eval_kwargs=dict(eval_horizon=500),
            parallelization="process",
            agent_name="PPO",
        ),
    )

    comparator.compare(manager1, manager2)
