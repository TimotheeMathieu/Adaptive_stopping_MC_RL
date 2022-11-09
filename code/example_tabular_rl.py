from gym_stochastic import TwentyOneWithDice
from maze_mdp import MazeMDPEnv
from ql_sarsa import QLAgent, SARSAAgent
from compare_agents import MultipleAgentsComparator
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
#21qvecDES
# manager1 = (
#     QLAgent,
#     dict(
#         train_env=(TwentyOneWithDice, {}),
#         fit_budget=int(1e4),
#         agent_name="QL",
#     ),
# )
# manager2 = (
#     SARSAAgent,
#     dict(
#         train_env=(TwentyOneWithDice, {}),
#         fit_budget=int(1e4),
#         agent_name="Sarsa",
#     ),
# )
#
# managers = [manager1, manager2]
#
# n = 5
# K = 4
# B=100_000
# alpha = 0.05
# n_managers = len(managers)
# print('With these parameters, we have a maximum of {} fits done for each agent'.format(n*K))
# print('Number of comparisons is {}'.format(n_managers*(n_managers-1)/2))
#
# comparator = MultipleAgentsComparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 100)
# comparator.compare(managers)
# plt.boxplot(comparator.eval_values, labels=['QL', 'SARSA'])
# plt.savefig("evals_21")
# print(comparator.mean_eval_values)
# plt.clf()
# comparator.plot_boundary()
# plt.savefig("compare_21")


env = MazeMDPEnv
manager1 = (
    QLAgent,
    dict(
        train_env=(env, {"height":4, "width": 4, "frac":0.2}),
        init_kwargs=dict(epsilon=0.3),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e4),
        agent_name="QL",
    ),
)
manager2 = (
    QLAgent,
    dict(
        train_env=(env, {"height":4, "width": 4, "frac":0.2}),
        init_kwargs=dict(epsilon=0.9),
        eval_kwargs=dict(eval_horizon=500),
        fit_budget=int(1e4),
        agent_name="QLeps",
    ),
)

managers = [manager1, manager2]

n = 5
K = 4
B=100_000
alpha = 0.05
n_managers = len(managers)
print('With these parameters, we have a maximum of {} fits done for each agent'.format(n*K))
print('Number of comparisons is {}'.format(n_managers*(n_managers-1)/2))

comparator = MultipleAgentsComparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 100)
comparator.compare(managers)
plt.boxplot(comparator.eval_values, labels=['QL', 'QLeps'])
plt.savefig("evals_maze")
print(comparator.mean_eval_values)
plt.clf()
comparator.plot_boundary()
plt.savefig("compare_maze")
