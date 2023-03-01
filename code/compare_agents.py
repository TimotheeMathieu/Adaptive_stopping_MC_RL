import logging
import numpy as np
from copy import copy
import os
from scipy import stats
from scipy.special import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

import rlberry
from rlberry.agents import Agent
from rlberry.envs import Model
import rlberry.spaces as spaces
from rlberry.manager import AgentManager
from rlberry.envs.interface import Model
from rlberry.seeding import Seeder
from rlberry.utils.writers import DefaultWriter

import itertools
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger()


class MultipleAgentsComparator:
    """
    Compare sequentially agents, with possible early stopping.
    At maximum, there can be n times K fits done.

    For now, implement only a two-sided test.

    Parameters
    ----------

    n: int, default=5
        number of fits before each early stopping check

    K: int, default=5
        number of check
    
    B: int, default=None
        Number of random permutations used to approximate permutation distribution.
    comparisons: list of tuple of indices or None
        if None, all the pairwise comparison are done.
        If = [(0,1), (0,2)] for instance, the compare only 0 vs 1  and 0 vs 2
    alpha: float, default=0.05
        level of the test

    beta: float, default=0
        power spent in early accept.

    n_evaluations: int, default=10
        number of evaluations used in the function _get_rewards.

    seed: int or None, default = None

    joblib_backend: str, default = "threading"
        backend to use to parallelize on multi-agents. Use "multiprocessing" or "loky" for a true parallelization.

    Attributes
    ----------

    decision: list of str in {"accept" , "reject"}
        decision of the tests for each comparison.

    rejected_decision: list of Tuple (comparison, bool)
       the choice of which agent is better in each rejected comparison


    Examples
    --------
    One can either use rlberry with self.compare, pre-computed scalars with self.compare_scalar or one can use
    the following code compatible with basically anything:

    >>> comparator = Comparator(n=6, K=6, B=10000, alpha=0.05, beta=0.01)
    >>>
    >>> Z = [np.array([]) for _ in agents]
    >>>
    >>> for k in range(comparator.K):
    >>>    for i, agent in enumerate(agents):
    >>>        # If the agent is still in one of the comparison considered, then generate new evaluations.
    >>>        if agent in comparator.current_comparisons.ravel():
    >>>            Z[i] = np.hstack([Z[i], train_evaluate(agent, n)])
    >>>    decisions, T = comparator.partial_compare(Z, verbose)
    >>>    if np.all([d in ["accept", "reject"] for d in decisions]):
    >>>        break

    Where train_evaluate(agent, n) is a function that trains n copies of agent and returns n evaluation values.
    """

    def __init__(
        self,
        n=5,
        K=5,
        B=10000,
        comparisons = None,
        alpha=0.05,
        beta=0,
        n_evaluations=1,
        seed=None,
        joblib_backend="threading",
    ):
        self.n = n
        self.K = K
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.n_evaluations = n_evaluations
        self.boundary = []
        self.test_stats = []
        self.k = 0
        self.level_spent = 0
        self.power_spent = 0
        self.seeder = Seeder(seed)
        self._writer = DefaultWriter("Comparator")
        self.rejected_decision = []
        self.joblib_backend = joblib_backend
        self.agent_names = []
        self.comparisons = comparisons
        self.current_comparisons = copy(comparisons)
        self.n_iters = None
        self.rng = None

    def compute_sum_diffs(self, k, Z, comparisons, boundary):
        """
        Compute the absolute value of the sum differences.
        """

        if k == 0:
            for _ in range(self.B):
                sum_diff = []
                for i, comp in enumerate(comparisons):
                    Zi = np.hstack([Z[comp[0]][: self.n], Z[comp[1]][: self.n]])
                    id_pos = self.rng.choice(2 * self.n, self.n, replace=False)
                    mask = np.zeros(2 * self.n)
                    mask[list(id_pos)] = 1
                    mask = mask == 1
                    sum_diff.append(np.sum(Zi[mask] - Zi[~mask]))
                self.sum_diffs.append(np.array(sum_diff))
        else:
            # Compute sums on B random permutations of blocks, conditional to not rejected.
            # Warning : there can be less than B resulting values due to rejection.

            # Eliminate for conditional
            sum_diffs = []
            for zval in self.sum_diffs:
                if np.max(zval) <= boundary[-1][1]:
                    sum_diffs.append(np.abs(zval))

            # add a new random permutation
            for j in range(len(self.sum_diffs)):
                id_pos = self.rng.choice(2 * self.n, self.n, replace=False)
                for i, comp in enumerate(comparisons):
                    Zk = np.hstack(
                        [
                            Z[comp[0]][(k * self.n) : ((k + 1) * self.n)],
                            Z[comp[1]][(k * self.n) : ((k + 1) * self.n)],
                        ]
                    )
                    mask = np.zeros(2 * self.n)
                    mask[list(id_pos)] = 1
                    mask = mask == 1
                    self.sum_diffs[j][i] += np.sum(Zk[mask] - Zk[~mask])

        return self.sum_diffs

    def partial_compare(self, Z, verbose=True):
        """
        Do the test of the k^th interim.

        Parameters
        ----------
        Z: array of size n_agents x k
            Concatenation All the evaluations all Agents till interim k
        verbose: bool
            print Steps
        Returns
        -------
        decision: str in {'accept', 'reject', 'continue'}
           decision of the test at this step.
        T: float
           Test statistic.
        bk: float
           thresholds.
        """
        if self.k == 0:
            # initialization
            n_managers = len(Z)
            if self.comparisons is None:
                self.comparisons = np.array(
                    [(i, j) for i in range(n_managers) for j in range(n_managers) if i < j]
                )
            self.current_comparisons = copy(self.comparisons)
            self.sum_diffs = []
            if self.n_iters is None:
                self.n_iters = [0] * n_managers
            self.decisions = np.array(["continue"] * len(self.comparisons))
            self.decisions_num = np.array([np.nan] * len(self.comparisons))
            self.id_tracked = np.arange(len(self.decisions))
            if self.rng is None:
                seeder = self.seeder.spawn(1)
                self.rng = seeder.rng
        k = self.k

        clevel = self.alpha*(k + 1) / self.K
        dlevel = self.beta*(k + 1) / self.K

        rs = np.abs(np.array(self.compute_sum_diffs(k, Z, self.current_comparisons, self.boundary)))

        if verbose:
            print("Step {}".format(k))

        current_decisions = self.decisions[self.decisions == "continue"]
        current_sign = np.zeros(len(current_decisions))
        
        for j in range(len(current_decisions)):
            rs_now = rs[:,current_decisions == "continue"]
            values = np.sort(
                np.max(rs_now, axis=1)
            )  # for now, don't care about ties. And there are ties, for instance when B is None, there are at least two of every values !

            icumulative_probas = np.arange(len(rs_now))[::-1] / self.B  # This corresponds to 1 - F(t) = P(T > t)

            # Compute admissible values, i.e. values that would not be rejected nor accepted.

            admissible_values_sup = values[
                self.level_spent + icumulative_probas <= clevel
            ]

            if len(admissible_values_sup) > 0:
                bk_sup = admissible_values_sup[0]  # the minimum admissible value
                level_to_add = icumulative_probas[
                    self.level_spent + icumulative_probas <= clevel
                ][0]
            else:
                # This case is possible if clevel-self.level_spent <= 1/self.B (smallest proba possible),
                # in which case there are not enough points and we don't take any decision for now. Happens in particular if B is None.
                bk_sup = np.inf
                level_to_add = 0

            cumulative_probas = np.arange(len(rs_now)) / self.B  # corresponds to P(T < t)
            admissible_values_inf = values[
                self.power_spent + cumulative_probas < dlevel
            ]

            if len(admissible_values_inf) > 0:
                bk_inf = admissible_values_inf[-1]  # the maximum admissible value
                power_to_add = cumulative_probas[
                    self.power_spent + cumulative_probas <= dlevel
                ][-1]
            else:
                bk_inf = -np.inf
                power_to_add = 0

            # Test statistic
            Tmax = 0
            Tmin = np.inf
            Tmaxsigned = 0
            Tminsigned = 0
            for i, comp in enumerate(self.current_comparisons[current_decisions == "continue"]):
                Ti = np.abs(
                    np.sum(
                        Z[comp[0]][: ((k + 1) * self.n)]
                        - Z[comp[1]][: ((k + 1) * self.n)]
                    )
                )
                if Ti > Tmax:
                    Tmax = Ti
                    imax = i
                    Tmaxsigned = np.sum(
                        Z[comp[0]][: ((k + 1) * self.n)]
                        - Z[comp[1]][: ((k + 1) * self.n)]
                    )

                if Ti < Tmin:
                    Tmin = Ti
                    imin = i
                    Tminsigned = np.sum(
                        Z[comp[0]][: ((k + 1) * self.n)]
                        - Z[comp[1]][: ((k + 1) * self.n)]
                    )

            if Tmax > bk_sup:
                id_reject = np.arange(len(current_decisions))[current_decisions== "continue"][imax]
                current_decisions[id_reject] = "reject"
                self.rejected_decision.append(self.current_comparisons[id_reject])
                current_sign[id_reject] = 2*(Tmaxsigned > 0)-1
                print("reject")
            elif Tmin < bk_inf:
                id_accept = np.arange(len(current_decisions))[current_decisions == "continue"][imin]
                current_decisions[id_accept] = "accept"
            else:
                break

            
        
        self.boundary.append((bk_inf, bk_sup))

        self.level_spent += level_to_add  # level effectively used at this point
        self.power_spent += power_to_add
        
        if k == self.K - 1:
            self.decisions[self.decisions == "continue"] = "accept"
        
        self.k = self.k + 1
        self.eval_values = Z
        self.mean_eval_values = [np.mean(z) for z in Z]
        self.n_iters = [len(z.ravel()) for z in Z]


        self.test_stats.append(Tmaxsigned)

        id_decided = np.array(current_decisions) != "continue"
        id_rejected = np.array(current_decisions) == "reject"
        id_accepted = np.array(current_decisions) == "accept"

        self.decisions[self.id_tracked[id_rejected]] = "reject"
        self.decisions[self.id_tracked[id_accepted]] = "accept"

        self.decisions_num[self.id_tracked] = current_sign

        self.id_tracked = self.id_tracked[~id_decided]
        self.current_comparisons = self.current_comparisons[~id_decided]
        self.sum_diffs = np.array(self.sum_diffs)[:, ~id_decided]
        return current_decisions, Tmaxsigned

    def compare(self, managers,  clean_after=True, verbose=True):
        """
        Compare the managers pair by pair.

        Parameters
        ----------
        managers : list of tuple of agent_class and init_kwargs for the agent.
        clean_after: boolean
        verbose: boolean
        """
        Z = [np.array([]) for _ in managers]
        # spawn independent seeds, one for each fit and one for the comparator.
        seeders = self.seeder.spawn(len(managers) * self.K + 1)
        self.rng = seeders[-1].rng
        
        for k in range(self.K):
            Z = self._fit(managers, Z, k, seeders, clean_after)
            current_decisions, T = self.partial_compare(Z, verbose)
            if np.all([d in ["accept", "reject"] for d in self.decisions]):
                break

        return self.decisions

    def compare_scalars(self, scalars):
        """
        Compare the managers pair by pair.
        Parameters
        ----------
        scalars : list of list of scalars.
        """
        Z = [np.array([]) for _ in scalars]


        for k in range(self.K):
            Z = self._get_z_scalars(scalars, Z, k)
            decisions, T = self.partial_compare(Z, k)
            if np.all([d in ["accept", "reject"] for d in self.decisions]):
                break
        return decisions

    def _get_z_scalars(self, scalars, Z, k):


        for i in range(len(scalars)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                Z[i] = np.hstack([Z[i], scalars[i][k * self.n : (k + 1) * self.n]])
        return Z

    def _fit(self, managers, Z, k, seeders, clean_after):
        agent_classes = [manager[0] for manager in managers]
        kwargs_list = [manager[1] for manager in managers]
        for kwarg in kwargs_list:
            kwarg["n_fit"] = self.n
        managers_in = []
        for i in range(len(agent_classes)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                agent_class = agent_classes[i]
                kwargs = kwargs_list[i]
                seeder = seeders[i]
                managers_in.append(AgentManager(agent_class, **kwargs, seed=seeder))
        if len(self.agent_names) == 0:
            self.agent_names = [m.agent_name for m in managers_in]
        # For now, paralellize only training because _get_evals not pickleable
        managers_in = Parallel(n_jobs=-1, backend=self.joblib_backend)(
            delayed(_fit_agent)(manager) for manager in managers_in
        )

        idz = 0
        for i in range(len(agent_classes)):
            if (self.current_comparisons is None) or (i in np.array(self.current_comparisons).ravel()):
                Z[i] = np.hstack([Z[i], self._get_evals(managers_in[idz])])
                idz += 1
        if clean_after:
            for m in managers_in:
                m.clear_output_dir()
        return Z

    def _get_evals(self, manager):
        """
        Can be overwritten for alternative evaluation function.
        """
        eval_values = []
        for idx in range(self.n):
            logger.info("Evaluating agent " + str(idx))
            eval_values.append(
                np.mean(manager.eval_agents(self.n_evaluations, agent_id=idx))
            )
        return eval_values


    def plot_boundary(self):
        """
        Graphical representation of the boundary and the test statistics as it was computed during the self.compare execution.
        self.compare must have been executed prior to executing plot_boundary.
        """
        assert len(self.boundary) > 0, "Boundary not found. Did you do the comparison?"

        y1 = np.array([b[0] for b in self.boundary])
        y2 = -y1
        y3 = np.array([b[1] for b in self.boundary])
        y4 = -y3

        # boundary plot
        x = np.arange(1, len(y1) + 1)
        p2 = plt.plot(x, y1, "o-", label="Boundary inf", alpha=0.7)
        plt.plot(x, y2, "o-", color=p2[0].get_color(), alpha=0.7)
        p3 = plt.plot(x, y3, "o-", label="Boundary sup", alpha=0.7)
        plt.plot(x, y4, "o-", color=p3[0].get_color(), alpha=0.7)

        # test stats plot
        for i, c in enumerate(self.comparisons):
            Ti = []
            Z1 = self.eval_values[c[0]]
            Z2 = self.eval_values[c[1]]

            K1 = self.n_iters[c[0]] // self.n
            K2 = self.n_iters[c[1]] // self.n

            for k in range(min(K1, K2)):
                T = np.sum(Z1[: ((k + 1) * self.n)]) - np.sum(Z2[: ((k + 1) * self.n)])
                if np.abs(T) <= self.boundary[k][1]:
                    Ti.append(
                        np.sum(Z1[: ((k + 1) * self.n)])
                        - np.sum(Z2[: ((k + 1) * self.n)])
                    )
                else:
                    Ti.append(
                        np.sum(Z1[: ((k + 1) * self.n)])
                        - np.sum(Z2[: ((k + 1) * self.n)])
                    )
                    break

            plt.scatter(x[: len(Ti)], Ti, label=str(c))

        plt.legend()

        plt.xlabel("$k$")
        plt.ylabel("test stat.")

    def plot_results(self, agent_names=None):
        """
        visual representation of results.
        """

        id_sort = np.argsort(self.mean_eval_values)
        Z = np.array(self.eval_values)[id_sort]

        if agent_names is None:
            agent_name = self.agent_names

        links = np.zeros([len(agent_names),len(agent_names)])
        for i in range(len(self.comparisons)):
            c = self.comparisons[i]
            links[c[0],c[1]] = self.decisions_num[i]
            
        links = links - links.T
        links = links[id_sort,:][:, id_sort]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 2], "hspace": 0}, figsize=(6,5)
        )
        the_table = ax1.table(
            cellText=[self.n_iters], rowLabels=["n_iter"], loc="top", cellLoc="center"
        )

        # Generate a custom colormap
        colors = np.array([(255, 80, 80), (102, 255, 102), (102, 153, 255)])/256
        cmap = LinearSegmentedColormap.from_list("my_cmap", colors, N=3)

        # Draw the heatmap with the mask and correct aspect ratio
        res = sns.heatmap(links,cmap=cmap, vmax=1, center=0,linewidths=.5, ax =ax1, 
                          cbar=False, yticklabels=np.array(agent_names)[id_sort])

        # Drawing the frame
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        ax2.boxplot(Z.T, labels=np.array(agent_names)[id_sort])
        # Creating legend with color box
        blue_patch = mpatches.Patch(color=colors[0], label='smaller')
        green_patch = mpatches.Patch(color=colors[1], label='equal')
        red_patch = mpatches.Patch(color=colors[2], label='larger')

        plt.legend(handles=[blue_patch, green_patch, red_patch],loc='center left', bbox_to_anchor=(1, 0.5))


def _fit_agent(manager):
    manager.fit()
    return manager
