
import numpy as np
import matplotlib.pyplot as plt

from compare_agents import MultipleAgentsComparator, Two_AgentsComparator

from rlberry.agents.torch import DQNAgent, PPOAgent, A2CAgent
from rlberry.agents import RSUCBVIAgent

from rlberry.envs import gym_make




n = 5 # Size of a group
K = 6 # number of groups
alpha = 0.05 # level of the test
B = 100000 # number of random permutation used to approximate the test


# # Comparison of two agents
# First, we verify that $n$ and $K$ are enough to detect the difference between two agents.
# Suppose that the two agents we want to compare are DQN and PPO on Acrobot.
#
# ## Power requirements
# From previous experiments, we know that the std of such agents on Acrobot is between 2 and 15 (see gym leaderboard) and we want to check that we can detect a difference of 5 between the rewards of the two agents




if __name__ == "__main__":

    comparator  = Two_AgentsComparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 100)
    print("Expected power is", comparator.power(M=100000, mup=0, muq = 5, sigmap=4, sigmaq=5))

    # n and K should be sufficient to detect the difference between the agents if the difference is more than 5.
    #
    # ## Environment and Agent definition

    env_ctor = gym_make
    env_kwargs = dict(id="Acrobot-v1")
    budget = 5e4

    manager1 = (
            DQNAgent,
            dict(
                train_env=(env_ctor, env_kwargs),
                fit_budget=budget,
                eval_kwargs=dict(eval_horizon=500),
                init_kwargs=dict(learning_rate=6.3e-4, batch_size=128, gamma=0.99),
                agent_name="DQN",
                parallelization="process"
            ),
            )
    manager2 = (
        PPOAgent,
        dict(
            train_env=(env_ctor, env_kwargs),
            fit_budget=budget,
            eval_kwargs=dict(eval_horizon=500),
            init_kwargs=dict(normalize_rewards=True, n_steps=256, gamma=0.99),
            agent_name="PPO",
            parallelization="process"
        ),
    )


    comparator.compare( manager2, manager1)
    # comparator.plot_boundary()
    print("Number of iterations used is "+str(comparator.n_iter))

    # # Multiple comparison
    # We want to compare with RSUCBVI


    manager3 = (
        RSUCBVI,
        dict(
            train_env=(env_ctor, env_kwargs),
            fit_budget=budget*10,
            eval_kwargs=dict(eval_horizon=500),
            init_kwargs=dict(gamma=0.99, horizon=300, bonus_scale_factor=0.01, min_dist=0.25),
            agent_name="RSUCBVI",
            parallelization="process"
        ),
    )
    managers = [manager1, manager2, manager3]
    comparator  = MultipleAgentsComparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 100)

    comparator.compare(managers)
    print(comparator.decisions,
          comparator.comparisons,
          comparator.rejected_decision)

    print('Evaluations are',comparator.eval_values)
