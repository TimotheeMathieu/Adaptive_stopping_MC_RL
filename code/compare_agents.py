import logging
import numpy as np
from copy import copy
import os
from scipy import stats
from scipy.special import binom
from scipy.stats import rankdata
import matplotlib.pyplot as plt

import rlberry
from rlberry.manager import AgentManager
from rlberry.envs.interface import Model
from rlberry.seeding import Seeder
from rlberry.utils.writers import DefaultWriter

import itertools


#logger = rlberry.logger
logger = logging.getLogger()

# TO DO:
# * At the last block, need only alpha*(2n choose n)**K statistics
# * Be careful about ties
# * Implement the Multiple Agent for multiple > 2


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

    def __init__(self, n=5, K=5, B=None, alpha=0.05, name="PK",  n_evaluations=1, seed=None):
        self.n = n
        self.K = K
        self.B = B
        self.alpha = alpha
        self.name = name
        self.n_evaluations = n_evaluations
        self.boundary = []
        self.test_stats = []
        self.level_spent = 0
        self.n_iter = 0
        self.seeder = Seeder(seed)
        self._writer = DefaultWriter("Comparator")

    def get_spending_fun(self):
        """
        Return the spending function corresponding to self.name

        should be an increasing function f such that f(0)=0 and f(1)=alpha
        """
        if self.name == "PK":
            return lambda p: self.alpha * np.log(1 + np.exp(1) * p - p)
        elif self.name == "OF":
            return lambda p: 2 - 2 * stats.norm.cdf(
                stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(p)
            )
        else:
            raise RuntimeError("name not implemented")


    def compute_sum_diffs(self,  k, Z, boundary):
        """
        Compute the absolute value of the sum differences.
        """
        Zk = Z[(k*2*self.n):((k+1)*2*self.n)]



        sum_diffs_k = []

        if self.B is None :
            # Enumerate all possibilities using previous enumeration
            # Pruning for conditional proba
            if k >0:
                idxs = np.where(np.array([r <= boundary[k-1] for r in self.sum_diffs]))[0]
                self.sum_diffs = [self.sum_diffs[i] for i in idxs]

            ids_pos = itertools.combinations(np.arange(2*self.n), self.n)
            # Compute all differences in current block
            for id_pos in ids_pos:
                mask = np.zeros(2*self.n)
                mask[list(id_pos)] = 1
                mask = mask == 1
                Tpos  = np.sum(Zk[mask]-Zk[~mask])
                sum_diffs_k.append(np.sum(Zk[mask]-Zk[~mask]))

            choices = np.arange(len(sum_diffs_k))

            # compute all the sum_diffs by combination.
            self.sum_diffs = np.array([ r + sum_diffs_k[int(choice)] for r in self.sum_diffs  for choice in choices], dtype=np.float32)

        else:
            # Compute sums on B random permutations of blocks, conditional to not rejected.
            # Warning : there can be less than B resulting values due to rejection.
            self.sum_diffs = []
            for _ in range(self.B):
                # sample a random permutation of all the blocks, conditional on not rejected till now.
                sum_diff = 0
                add_it = True
                for j in range(k+1):
                    Zj = Z[(j*2*self.n):((j+1)*2*self.n)]
                    mask = np.zeros(2*self.n)
                    id_pos = self.rng.choice(2*self.n, self.n, replace=False)
                    mask[list(id_pos)] = 1
                    mask = mask == 1
                    sum_diff += np.sum(Zj[mask]-Zj[~mask])
                    if (j < k) and sum_diff > boundary[j]:
                        add_it = False
                        break
                if add_it :
                    self.sum_diffs.append(sum_diff)

        return self.sum_diffs



    def partial_compare(self, Z, X, k):
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

        Returns
        -------
        decision: str in {'accept', 'reject', 'continue'}
           decision of the test at this step.
        T: float
           Test statistic.
        bk: gloat
           threshold.


        """
        spending_fun = self.get_spending_fun()

        clevel = spending_fun((k + 1) / self.K)

        rs = np.abs(np.array(self.compute_sum_diffs( k, Z, self.boundary)))

        values = np.sort(rs) # for now, don't care about ties. And there are ties, for instance when B is None, there are at least two of every values !

        icumulative_probas = np.arange(len(rs))[::-1]/len(rs) # This corresponds to 1 - F(t) = P(T > t)

        # Compute admissible values, i.e. values that would not be rejected.
        admissible_values_sup = values[
            self.level_spent + icumulative_probas <= clevel
        ]

        if len(admissible_values_sup)>0:
            bk = admissible_values_sup[0]  # the minimum admissible value
            level_to_add = icumulative_probas[self.level_spent + icumulative_probas <= clevel][0]
        else:
            # This case is possible if clevel-self.level_spent <= 1/len(rs) (smallest proba possible),
            # in which case there are not enough points and we don't take any decision for now. Happens in particular if B is None.
            bk = np.inf
            level_to_add = 0

        # Test statistic
        T = np.abs(np.sum(Z*(-1)**X))

        # p-value computation. Not used in the algo, just for info purpose.
        if len(icumulative_probas[values >= T])>0:
            p_value = self.level_spent + icumulative_probas[values >= T][0]
        else:
            p_value = self.level_spent

        self.level_spent += level_to_add # level effectively used at this point

        self.boundary.append(bk)

        self._writer.add_scalar("Stat_val", T, k)
        self._writer.add_scalar("sup_bound", bk, k)

        if (T > bk):
            decision = "reject"
        elif k == self.K - 1:
            decision = "accept"
        else:
            decision = "continue"

        return decision, np.sum(Z*(-1)**X), bk, p_value

    def compare(self, manager1, manager2):
        """
        Compare manager1 and manager2 performances

        Parameters
        ----------
        manager1 : tuple of agent_class and init_kwargs for the agent.
        manager2 : tuple of agent_class and init_kwargs for the agent.

        """
        X = np.array([])
        Z = np.array([])

        agent_class1 = manager1[0]
        kwargs1 = manager1[1]
        kwargs1["n_fit"] = self.n

        agent_class2 = manager1[0]
        kwargs2 = manager2[1]
        kwargs2["n_fit"] = self.n

        Z = np.array([]) # concatenated values of evaluation
        X = np.array([]) # assignement vector of 0 and 1

        # Initialization of the permutation distribution
        self.sum_diffs = [0]

        # spawn independent seeds, one for each fit and one for the comparator.
        seeders = self.seeder.spawn(2*self.K+1)
        self.rng = seeders[-1].rng

        for k in range(self.K):

            m1 = AgentManager(agent_class1, **kwargs1, seed=seeders[2*k])
            m2 = AgentManager(agent_class2, **kwargs2, seed=seeders[2*k+1])

            m1.fit()
            m2.fit()
            self.n_iter += 2*self.n

            Z = np.hstack([Z, self._get_evals(m1), self._get_evals(m2)])
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])

            self.decision, Tsigned, bk, p_val = self.partial_compare(Z, X, k)

            self.test_stats.append(Tsigned)

            if self.decision == "reject":
                logger.info("Reject the null after " + str(k + 1) + " groups")
                if Tsigned <= 0:
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

        self.p_val = p_val
        logger.info('p value is '+str(p_val))

    def plot_boundary(self):
        """
        Graphical representation of the boundary and the test statistics as it was computed during the self.compare execution.

        self.compare must have been executed prior to executing plot_boundary.
        """
        assert len(self.boundary)>0, "Boundary not found. Did you do execute the comparison?"

        y1 = np.array(self.boundary)
        y2 = -y1

        x = np.arange(1, len(y1) + 1)
        p2 = plt.plot(x, y1, "o-", label="Boundary", alpha=0.7)
        plt.plot(x, y2, "o-", color=p2[0].get_color(), alpha=0.7)
        plt.scatter(x, self.test_stats, color="red", label="observations")
        plt.legend()
        plt.xlabel('$k$')
        plt.ylabel('test stat.')



    def _get_evals(self, manager):
        """
        Can be overwritten for alternative evaluation function.
        """
        eval_values = []
        for idx in range(self.n):
            logger.info("Evaluating agent " + str(idx))
            eval_values.append(
                np.mean(
                    manager.eval_agents(self.n_evaluations, agent_id=idx)
                )
            )
        return eval_values


# Draft, still todo.
class MultipleAgentsComparator:
    """
    Compare sequentially two agents, with possible early stopping.
    At maximum, there can be n times K fits done.

    For now, implement only a two-sided test.

    !! STILL NOT FINISHED !! For now, use TwoAgentsComparator

    Parameters
    ----------

    n: int, default=5
        number of fits before each early stopping check

    K: int, default=5
        number of checks

    alpha: float, default=0.05
        level of the test

    name: str in {'PK', 'OF'}, default = "PK"
        type of spending function to use.

    n_evaluations: int, default=10
        number of evaluations used in the function _get_rewards.

    seed: int or None, default = None

    Attributes
    ----------

    decision: str in {"accept" , "reject"}
        decision of the test.

    p_val: float
        p-value of the test
    """

    def __init__(self, n=5, K=5, alpha=0.05, name="PK",  n_evaluations=1, seed=None):
        self.n = n
        self.K = K
        self.alpha = alpha
        self.name = name
        self.ttype = ttype
        self.n_evaluations = n_evaluations
        self.seed = seed

    def compare(self, managers):
        """
        Compare the managers pair by pair using Bonferroni correction.

        Parameters
        ----------
        managers : list of tuple of agent_class and init_kwargs for the agent.

        """
        pairs = list(itertools.combinations(managers),2)
        level = self.alpha / len(pairs)

        decisions = []
        for pair in pairs:
            comparator = Two_AgentsComparator(self.n, self.K, self.alpha, self.name, self.n_evaluations, self.seed)
            comparator.compare(pair[0], pair[1])
            decisions.append((comparator.agent1_name, comparator.agent2_name), comparator.decision)

        return decisions