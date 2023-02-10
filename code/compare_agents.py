import logging
import numpy as np
from copy import copy
import os
from scipy import stats
from scipy.special import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

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


# TO DO:
# * Be careful about ties


class Two_AgentsComparator:
    """
    Compare sequentially two agents, with possible early stopping.
    At maximum, there can be n times K fits done.
    For now, implement only a two-sided test.
    Parameters
    ----------
    n: int, default=5
        number of fits before each early stopping check
    K: int, default=5
        number of checks
    B: int or None, default=None
        number of random permutation used in each block to approximate permutation distribution.
        If not None, the final complexity is K*B*n^2.
    alpha: float, default=0.05
        level of the test
    beta: float, default=0
        power spent in early accept.
    name: str in {'PK', 'OF'}, default = "PK"
        type of spending function to use.
    n_evaluations: int, default=10
        number of evaluations used in the function _get_rewards.
    seed: int or None, default = None
    Attributes
    ----------
    decision: str in {"accept" , "reject"}
        decision of the test.
    boundary: array of size at most K
        Boundary computed during the test.
    n_iter: int
        Total number of fits used.
    eval_1: array of floats
        Evaluations of Agent 1 used during comparison
    eval_2: array of floats
        Evaluations of Agent 1 used during comparison
    agent1_name: str
        Name of Agent 1
    agent2_name: str
        Name of Agent 2
    test_stats: array of floats
        Value of the statistic T = sum(Eval_agent1)-sum(Eval_agent2) at each interim.
    """

    def __init__(
        self, n=5, K=5, B=None, alpha=0.05, beta=0, name="PK", n_evaluations=1, seed=None
    ):
        self.n = n
        self.K = K
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.n_evaluations = n_evaluations
        self.boundary = []
        self.test_stats = []
        self.level_spent = 0
        self.power_spent = 0
        self.n_iter = 0
        self.seeder = Seeder(seed)
        self._writer = DefaultWriter("Comparator")

    def get_spending_fun(self, level):
        """
        Return the spending function corresponding to self.name
        should be an increasing function f such that f(0)=0 and f(1)=alpha
        """
        if self.name == "PK":
            return lambda p: level * np.log(1 + np.exp(1) * p - p)
        elif self.name == "OF":
            return lambda p: 2 - 2 * stats.norm.cdf(
                stats.norm.ppf(1 - level / 2) / np.sqrt(p)
            )
        else:
            raise RuntimeError("name not implemented")

    def compute_sum_diffs(self, k, Z, boundary):
        """
        Compute the absolute value of the sum differences.
        """
        Zk = Z[(k * 2 * self.n) : ((k + 1) * 2 * self.n)]

        sum_diffs_k = []

        if self.B is None:
            raise ValueError('B is None not maintained anymore')
            # Enumerate all possibilities using previous enumeration
            # Pruning for conditional proba
            if k > 0:
                idxs = np.where(
                    np.array([boundary[k - 1][0]<= r <= boundary[k - 1][1] for r in self.sum_diffs])
                )[0]
                self.sum_diffs = [self.sum_diffs[i] for i in idxs]

            ids_pos = itertools.combinations(np.arange(2 * self.n), self.n)
            # Compute all differences in current block
            for id_pos in ids_pos:
                mask = np.zeros(2 * self.n)
                mask[list(id_pos)] = 1
                mask = mask == 1
                Tpos = np.sum(Zk[mask] - Zk[~mask])
                sum_diffs_k.append(np.sum(Zk[mask] - Zk[~mask]))

            choices = np.arange(len(sum_diffs_k))

            # compute all the sum_diffs by combination.
            self.sum_diffs = np.array(
                [
                    r + sum_diffs_k[int(choice)]
                    for r in self.sum_diffs
                    for choice in choices
                ],
                dtype=np.float32,
            )

        else:
            # Compute sums on B random permutations of blocks, conditional to not rejected.
            # Warning : there can be less than B resulting values due to rejection.
            self.sum_diffs = []
            for _ in range(self.B):
                # sample a random permutation of all the blocks, conditional on not rejected/accepted till now.
                sum_diff = 0
                add_it = True
                for j in range(k + 1):
                    Zj = Z[(j * 2 * self.n) : ((j + 1) * 2 * self.n)]
                    mask = np.zeros(2 * self.n)
                    id_pos = self.rng.choice(2 * self.n, self.n, replace=False)
                    mask[list(id_pos)] = 1
                    mask = mask == 1
                    sum_diff += np.sum(Zj[mask] - Zj[~mask])
                    if (j < k) and ((sum_diff > boundary[j][1]) or (sum_diff< boundary[j][0])):
                        add_it = False
                        break
                if add_it:
                    self.sum_diffs.append(sum_diff)

        return self.sum_diffs

    def partial_compare(self, Z, X, k, verbose=True):
        """
        Do the test of the k^th interim.
        Parameters
        ----------
        Z: array of size 2*self.n*(k+1)
            Concatenation All the evaluations of Agent 1 and Agent2 up till interim k
        X: array of size 2*self.n*(k+1)
            {0,1} array with the affectation of each value of Z to either Agent 1 (0) or Agent 2 (1)
        k: int
            index of the interim, in {0,...,K-1}
        verbose: boolean
        Returns
        -------
        decision: str in {'accept', 'reject', 'continue'}
           decision of the test at this step.
        T: float
           Test statistic.
        bk: gloat
           threshold.
        """
        spending_fun_a = self.get_spending_fun(self.alpha)
        spending_fun_b = self.get_spending_fun(self.beta)

        clevel = spending_fun_a((k + 1) / self.K)
        dlevel = spending_fun_b((k + 1) / self.K)

        rs = np.abs(np.array(self.compute_sum_diffs(k, Z, self.boundary)))

        values = np.sort(
            rs
        )  # for now, don't care about ties. And there are ties, for instance when B is None, there are at least two of every values !

        icumulative_probas = np.arange(len(rs))[::-1] / self.B
        # This corresponds to 1 - F(t) = P(T > t)

        # Compute admissible values, i.e. values that would not be rejected.
        admissible_values_sup = values[self.level_spent + icumulative_probas <= clevel]

        if len(admissible_values_sup) > 0:
            bk_sup = admissible_values_sup[0]  # the minimum admissible value
            level_to_add = icumulative_probas[
                self.level_spent + icumulative_probas <= clevel
            ][0]
        else:
            # This case is possible if clevel-self.level_spent <= 1/len(rs) (smallest proba possible),
            # in which case there are not enough points and we don't take any decision for now. Happens in particular if B is None.
            bk_sup = np.inf
            level_to_add = 0

        cumulative_probas = np.arange(len(rs)) / self.B # corresponds to P(T < t)
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
        T = np.abs(np.sum(Z * (-1) ** X))

        self.level_spent += level_to_add  # level effectively used at this point
        self.power_spent += power_to_add

        self.boundary.append((bk_inf, bk_sup))

        self._writer.add_scalar("Stat_val", T, k)
        self._writer.add_scalar("sup_bound", bk_sup, k)
        self._writer.add_scalar("inf_bound", bk_inf, k)

        if T > bk_sup:
            decision = "reject"
        elif T < bk_inf:
            decision = "accept"
        elif k == self.K - 1:
            decision = "accept"
        else:
            decision = "continue"

        return decision, np.sum(Z * (-1) ** X), bk_inf, bk_sup

    def compare(self, manager1, manager2, clean_after=True, verbose=True):
        """
        Compare manager1 and manager2 performances
        Parameters
        ----------
        manager1 : tuple of agent_class and init_kwargs for the agent.
        manager2 : tuple of agent_class and init_kwargs for the agent.
        clean_after: boolean
        verbose: boolean
        """
        X = np.array([])
        Z = np.array([])

        agent_class1 = manager1[0]
        kwargs1 = manager1[1]
        kwargs1["n_fit"] = self.n

        agent_class2 = manager2[0]
        kwargs2 = manager2[1]
        kwargs2["n_fit"] = self.n

        Z = np.array([])  # concatenated values of evaluation
        X = np.array([])  # assignement vector of 0 and 1

        # Initialization of the permutation distribution
        self.sum_diffs = [0]

        # spawn independent seeds, one for each fit and one for the comparator.
        seeders = self.seeder.spawn(2 * self.K + 1)
        self.rng = seeders[-1].rng

        for k in range(self.K):

            m1 = AgentManager(agent_class1, **kwargs1, seed=seeders[2 * k])
            m2 = AgentManager(agent_class2, **kwargs2, seed=seeders[2 * k + 1])

            m1.fit()
            m2.fit()
            self.n_iter += 2 * self.n

            Z = np.hstack([Z, self._get_evals(m1), self._get_evals(m2)])
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])

            self.decision, Tsigned, bk_inf, bk_sup = self.partial_compare(
                Z, X, k, verbose
            )

            self.test_stats.append(Tsigned)
            if clean_after:
                m1.clear_output_dir()
                m2.clear_output_dir()
            if self.decision == "reject":
                logger.info("Reject the null after " + str(k + 1) + " groups")
                if Tsigned >= 0:
                    logger.info(m1.agent_name + " is better than " + m2.agent_name)
                else:
                    logger.info(m2.agent_name + " is better than " + m1.agent_name)

                break
            else:
                logger.info("Did not reject on interim " + str(k + 1))
        if self.decision == "accept":
            logger.info(
                "Did not reject the null hypothesis: either K, n are too small or the agents perform similarly"
            )

        self.eval_1 = Z[X == 0]
        self.eval_2 = Z[X == 1]
        self.agent1_name = m1.agent_name
        self.agent2_name = m2.agent_name
        self.eval_values = Z
        self.affectations = X


    def compare_scalars(self, scalar_list1, scalar_list2, names=["SAC", "TD3"]):
        """
        Compare two lists of performances sclaras
        Parameters
        ----------
        scalar_list1 : list of evaluation means of an agent fitted len(scalar_list1) times.
        scalar_list2 : list of evaluation means of an agent fitted len(scalar_list2) times.
        clean_after: boolean
        """
        assert len(scalar_list1) == len(
            scalar_list2
        ), "Scalar lists should have same size."

        Z = np.array([])  # concatenated values of evaluation
        X = np.array([])  # assignement vector of 0 and 1

        # Initialization of the permutation distribution
        self.sum_diffs = [0]

        # spawn independent seeds, one for each fit and one for the comparator.
        seeders = self.seeder.spawn(2 * self.K + 1)
        self.rng = seeders[-1].rng

        for k in range(self.K):
            self.n_iter += 2 * self.n
            print(self.n_iter)

            Z = np.hstack(
                [
                    Z,
                    scalar_list1[k * self.n : (k + 1) * self.n],
                    scalar_list2[k * self.n : (k + 1) * self.n],
                ]
            )
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])

            self.decision, Tsigned, bk_inf, bk_sup = self.partial_compare(
                Z, X, k
            )

            self.test_stats.append(Tsigned)
            if self.decision == "reject":
                logger.info("Reject the null after " + str(k + 1) + " groups")
                if Tsigned <= 0:
                    print(names[0] + " is better than " + names[1])
                else:
                    print(names[1] + " is better than " + names[0])

                break
            else:
                logger.info("Did not reject on interim " + str(k + 1))
        if self.decision == "accept":
            logger.info(
                "Did not reject the null hypothesis: either K, n are too small or the agents perform similarly"
            )

        self.eval_1 = Z[X == 0]
        self.eval_2 = Z[X == 1]
        self.agent1_name = names[0]
        self.agent2_name = names[1]
        self.eval_values = Z
        self.affectations = X


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

        plt.scatter(x, self.test_stats, color="red", label="observations")
        plt.legend()
        plt.xlabel("$k$")
        plt.ylabel("test stat.")

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

    def asym_boundary(self, M=100000):
        """
        Compute the asymptotic boundary of b_1,...,b_K using M Monte-Carlo approximation, with \tau(P,Q)=1.

        M is int, number of MC used to approximate
        """
        W = np.random.normal(size=(self.K, M))

        # computation of b_1 is exact and don't need Monte-Carlo approximation
        spending_fun = self.get_spending_fun()

        alphas = [spending_fun((k + 1) / self.K) for k in range(self.K)]
        boundary = [norm.ppf(1 - alphas[0])]

        # computation of b_k, k > 1
        for k in range(1, self.K):
            res = 0
            values = []
            for m in range(M):
                # condition on the past
                event = np.all(
                    [np.abs(np.mean(W[: (j + 1), m])) < boundary[j] for j in range(k)]
                )
                if event:
                    values.append(np.abs(np.mean(W[: (k + 1), m])))
            boundary.append(np.quantile(values, 1 - (alphas[k] - alphas[k - 1])))

        return boundary

    def power(self, M, mup, muq, sigmap, sigmaq):
        """
        Estimation of the power using M MC estimation. mup, muq are the mean of the two agents and sigmap, sigmaq are their respective stds.
        """

        W = np.random.normal(size=(self.K, M))
        boundary = self.asym_boundary(M)

        delta = np.abs(mup - muq) / np.sqrt(sigmap**2 + sigmaq**2)
        boundary_factor = np.sqrt(
            sigmap**2 + sigmaq**2 + (mup - muq) ** 2 / 2
        ) / np.sqrt(sigmap**2 + sigmaq**2)

        events = []
        for m in tqdm(range(M)):
            events.append(
                np.any(
                    [
                        np.abs(np.mean(W[: (j + 1), m]) + np.sqrt(self.n) * delta)
                        > boundary[j] * boundary_factor
                        for j in range(self.K)
                    ]
                )
            )
        return np.mean(events)


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

    name: str in {'PK', 'OF'}, default = "PK"
        type of spending function to use.

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
        B=None,
        comparisons = None,
        alpha=0.05,
        beta=0,
        name="PK",
        n_evaluations=1,
        seed=None,
        joblib_backend="threading",
    ):
        self.n = n
        self.K = K
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.n_evaluations = n_evaluations
        self.boundary = []
        self.test_stats = []
        self.k = 0
        self.level_spent = 0
        self.power_spent = 0
        self.seeder = Seeder(seed)
        self._writer = DefaultWriter("Comparator")
        self.rejected_decision = []
        self.rejected_sign = []
        self.joblib_backend = joblib_backend
        self.agent_names = []
        self.comparisons = comparisons
        self.current_comparisons = copy(comparisons)
        self.n_iters = None

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
            self.id_tracked = np.arange(len(self.decisions))
        k = self.k
        spending_fun_a = self.get_spending_fun(self.alpha)
        spending_fun_b = self.get_spending_fun(self.beta)

        clevel = spending_fun_a((k + 1) / self.K)
        dlevel = spending_fun_b((k + 1) / self.K)

        rs = np.abs(np.array(self.compute_sum_diffs(k, Z, self.current_comparisons, self.boundary)))

        if verbose:
            print("Step {}".format(k))

        current_decisions = self.decisions[self.decisions == "continue"]

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
                self.rejected_sign.append(Tmaxsigned > 0)
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

        self.id_tracked = self.id_tracked[~id_decided]
        self.current_comparisons = self.current_comparisons[~id_decided]
        self.sum_diffs = np.array(self.sum_diffs)[:, ~id_decided]
        return current_decisions, Tmaxsigned

    def compare(self, managers,  clean_after=True, verbose=True):
        """
        Compare the managers pair by pair using Bonferroni correction.

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
        Compare the managers pair by pair using Bonferroni correction.
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
        return 

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

    def get_spending_fun(self, level):
        """
        Return the spending function corresponding to self.name
        should be an increasing function f such that f(0)=0 and f(1)=level
        """
        if self.name == "PK":
            return lambda p: level * np.log(1 + np.exp(1) * p - p)
        elif self.name == "OF":
            return lambda p: 2 - 2 * stats.norm.cdf(
                stats.norm.ppf(1 - level / 2) / np.sqrt(p)
            )
        else:
            raise RuntimeError("name not implemented")

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
            if self.decisions[i] == "accept":
                links[c[0],c[1]] = -1
            else:
                links[c[0],c[1]] = 1
        links = links + links.T
        links = links[id_sort,:][:, id_sort]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 2], "hspace": 0}, figsize=(6,5)
        )
        the_table = ax1.table(
            cellText=[self.n_iters], rowLabels=["n_iter"], loc="top", cellLoc="center"
        )

        # Generate a mask for the upper triangle
        #mask = np.triu(np.ones_like(links, dtype=bool))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        res = sns.heatmap(links,cmap=cmap, vmax=1, center=0,linewidths=.5, ax =ax1, 
                          cbar=False, yticklabels=np.array(agent_names)[id_sort])

        # Drawing the frame
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        ax2.boxplot(Z.T, labels=np.array(agent_names)[id_sort])


def _fit_agent(manager):
    manager.fit()
    return manager
