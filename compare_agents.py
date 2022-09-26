import logging
import numpy as np
from copy import copy
import os
from scipy import stats
from scipy.special import binom
from scipy.stats import rankdata

import pdb
from sortedcontainers import SortedList
from rlberry.manager import AgentManager
from rlberry.envs.interface import Model
from rlberry.seeding import Seeder
from rlberry.utils.writers import DefaultWriter

import itertools

import rlberry

#logger = rlberry.logger
logger = logging.getLogger()

# SMART THINGS TO DO:
# * At the last block, need only alpha*(2n choose n)**K statistics


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
        self.level_spent1 = 0
        self.level_spent2 = 0
        self.seeder = Seeder(seed)
        self._writer = DefaultWriter("Comparator")

    def get_spending_fun(self):
        """
        Return the spending function corresponding to self.name
        """
        if self.name == "PK":
            return lambda p: self.alpha * np.log(1 + np.exp(1) * p - p)
        elif self.name == "OF":
            return lambda p: 2 - 2 * stats.norm.cdf(
                stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(p)
            )
        else:
            raise RuntimeError("name not implemented")


    def compute_means_diffs(self,  k, Z, boundary):
        """
        Compute the absolute value of the mean differences.
        """
        Zk = Z[(k*2*self.n):((k+1)*2*self.n)]

        # Pruning for conditional proba
        if k >0:
            idxs = np.where(np.array([r <= boundary[k-1] for r in self.mean_diffs]))[0]
            self.mean_diffs = [self.mean_diffs[i] for i in idxs]

        mean_diffs_k = []
        # TODO : do everything directly with absolute value and do also the test on the absolute value. Simpler.

        if self.B is None :
            ids_pos = itertools.combinations(np.arange(2*self.n), self.n)
            # Compute all differences in current block
            for id_pos in ids_pos:
                mask = np.zeros(2*self.n)
                mask[list(id_pos)] = 1
                mask = mask == 1
                Tpos  = np.sum(Zk[mask]-Zk[~mask])
                mean_diffs_k.append(np.sum(Zk[mask]-Zk[~mask]))
                mean_diffs_k = np.unique(np.abs(np.array(mean_diffs_k)))
                choices = np.arange(len(mean_diffs_k))

                # compute all the mean_diffs by combination.
                self.mean_diffs = np.array([ r + mean_diffs_k[int(choice)] for r in self.mean_diffs  for choice in choices], dtype=np.float32)
        else:
            self.mean_diffs = []
            for _ in range(self.B):
                # sample a random permutation of all the blocks, conditional on not rejected till now.
                mean_diff = 0
                add_it = True
                for j in range(k+1):
                    Zj = Z[(j*2*self.n):((j+1)*2*self.n)]
                    mask = np.zeros(2*self.n)
                    id_pos = self.rng.choice(2*self.n, self.n, replace=False)
                    mask[list(id_pos)] = 1
                    mask = mask == 1
                    mean_diff += np.sum(Zj[mask]-Zj[~mask])
                    if (j < k) and mean_diff > boundary[j]:
                        add_it = False
                        break
                if add_it :
                    self.mean_diffs.append(mean_diff)

            self.mean_diffs = np.abs(np.array(self.mean_diffs))



        return self.mean_diffs



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

        rs = self.compute_means_diffs( k, Z, self.boundary)
        probas = np.ones(len(rs)) / len(rs)

        idx = np.argsort(rs)
        values = np.array(rs)[idx]

        icumulative_probas = np.sum(probas) - np.cumsum(probas[idx])
        admissible_values_sup = values[
            self.level_spent1 + icumulative_probas <= clevel
        ]

        if len(admissible_values_sup)>0:
            bk_sup = admissible_values_sup[0]  # the minimum admissible value
            level_to_add1 = icumulative_probas[self.level_spent1 + icumulative_probas <= clevel][0]
        else:
            bk_sup = np.inf
            level_to_add1 = 0

        T = np.abs(np.sum(Z*(-1)**X))

        if len(icumulative_probas[values >= T])>0:
            p_value = self.level_spent1 + icumulative_probas[values >= T][0]
        else:
            p_value = self.level_spent1

        self.level_spent1 += level_to_add1

        self.boundary.append(bk_sup)

        self._writer.add_scalar("Stat_val", T, k)
        self._writer.add_scalar("sup_bound", bk_sup, k)

        if (T > bk_sup):
            decision = "reject"
        elif k == self.K - 1:
            decision = "accept"
        else:
            decision = "continue"

        return decision, np.sum(Z*(-1)**X), bk_sup, p_value

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

        Z = np.array([])
        X = np.array([])

        self.level_spent1 = 0
        self.level_spent2 = 0

        self.mean_diffs = [0]

        seeders = self.seeder.spawn(2*self.K+1)
        self.rng = seeders[-1].rng
        for k in range(self.K):

            m1 = AgentManager(agent_class1, **kwargs1, seed=seeders[2*k])
            m2 = AgentManager(agent_class2, **kwargs2, seed=seeders[2*k+1])

            m1.fit()
            m2.fit()

            Z = np.hstack([Z, self._get_evals(m1), self._get_evals(m2)])
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])


            self.decision, Tsigned, bk_sup, p_val = self.partial_compare(Z, X, k)


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
