from gym_stochastic import TwentyOneWithDice
from maze_mdp import MazeMDPEnv
from ql_sarsa import QLAgent, SARSAAgent
from compare_agents import MultipleAgentsComparator
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit

env = MazeMDPEnv
manager1 = (
    QLAgent,
    dict(
        train_env=(env, {}),
        init_kwargs=dict(epsilon=0.3),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e5),
        agent_name="QL-eps0.3",
    ),
)
manager2 = (
    SARSAAgent,
    dict(
        train_env=(env, {} ),
        init_kwargs=dict(epsilon=0.3),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e5),
        agent_name="SARSA-eps0.3",
    ),
)
manager3 = (
    QLAgent,
    dict(
        train_env=(env, {}),
        init_kwargs=dict(epsilon=0.1),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e5),
        agent_name="QL-eps0.1",
    ),
)
manager4 = (
    SARSAAgent,
    dict(
        train_env=(env, {} ),
        init_kwargs=dict(epsilon=0.1),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e5),
        agent_name="SARSA-eps0.1",
    ),
)
manager5 = (
    QLAgent,
    dict(
        train_env=(env, {}),
        init_kwargs=dict(epsilon=0.5),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e5),
        agent_name="QL-eps0.5",
    ),
)
manager6 = (
    SARSAAgent,
    dict(
        train_env=(env, {} ),
        init_kwargs=dict(epsilon=0.5),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e5),
        agent_name="SARSA-eps0.5",
    ),
)
managers = [manager1,manager2,manager3,manager4,manager5,manager6]

n = 5
K = 4
B=100_000
alpha = 0.05
n_managers = len(managers)
print('With these parameters, we have a maximum of {} fits done for each agent'.format(n*K))
print('Number of comparisons is {}'.format(n_managers*(n_managers-1)/2))

comparator = MultipleAgentsComparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 100)
comparator.compare(managers)
plt.boxplot(comparator.eval_values, labels=['QL-eps0.3', 'SARSA-eps0.3', 'QL-eps0.1', 'SARSA-eps0.1', 'QL-eps0.5', 'SARSA-eps0.5'])
plt.savefig("evals_maze")
print(comparator.mean_eval_values)
plt.clf()
comparator.plot_boundary()
plt.savefig("compare_maze")
