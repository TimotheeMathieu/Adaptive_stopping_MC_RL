from gym_stochastic import TwentyOneWithDice
from ql_sarsa import QLAgent, SARSAAgent
from compare_agents import MultipleAgentsComparator
import numpy as np
import matplotlib.pyplot as plt

manager1 = (
    QLAgent,
    dict(
        train_env=(TwentyOneWithDice, {}),
        fit_budget=int(1e4),
        agent_name="QL",
    ),
)
manager2 = (
    SARSAAgent,
    dict(
        train_env=(TwentyOneWithDice, {}),
        fit_budget=int(1e4),
        agent_name="Sarsa",
    ),
)

managers = [manager1, manager2]

n = 10
K = 10
B=100_000
alpha = 0.05
n_managers = len(managers)
print('With these parameters, we have a maximum of {} fits done for each agent'.format(n*K))
print('Number of comparisons is {}'.format(n_managers*(n_managers-1)/2))

class Comparator(MultipleAgentsComparator):
    def __init__(self, **kwargs):
        MultipleAgentsComparator.__init__(self, **kwargs)

    def _get_evals(self, manager):
        """
        Compute the cumulative reward.
        """
        eval_values = []
        for idx in  manager.get_writer_data():
            df = manager.get_writer_data()[idx]
            eval_values.append(np.sum(df.loc[df['tag']=='reward', 'value']))
        return eval_values

comparator = Comparator(n=n, K=K, B=B, alpha=alpha, n_evaluations = 100)
comparator.compare(managers)
plt.boxplot(comparator.eval_values, labels=['QL', 'SARSA'])
plt.show()
print(comparator.mean_eval_values)
comparator.plot_boundary()
