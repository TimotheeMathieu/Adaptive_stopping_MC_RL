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

    def __init__(self, n=5, K=5, alpha=0.05, name="PK",  n_evaluations=1, seed=None):
        self.n = n
        self.K = K
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
        Compute the means differences.
        """
        Zk = Z[(k*2*self.n):((k+1)*2*self.n)]
        ids_pos = itertools.combinations(np.arange(2*self.n), self.n)
        id_md = 0

        # Pruning for conditional proba
        if k >0:
            idxs = np.where(np.array([boundary[k-1][0] <= r <= boundary[k-1][1] for r in self.mean_diffs]))[0]
            self.mean_diffs = [self.mean_diffs[i] for i in idxs]

        mean_diffs_k = []
        for id_pos in ids_pos:
            mask = np.zeros(2*self.n)
            mask[list(id_pos)] = 1
            mask = mask == 1
            mean_diffs_k.append(np.sum(Zk[mask]-Zk[~mask]))
            id_md += 1


        choices = np.arange(int(binom(2*self.n, self.n)))

        self.mean_diffs = np.array([ r + mean_diffs_k[choice] for r in self.mean_diffs  for choice in choices])
        # very inefficient. To do : improve computation.
            
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
            self.level_spent1 + icumulative_probas <= clevel / 2
        ]

        if len(admissible_values_sup)>0:
            bk_sup = admissible_values_sup[0]  # the minimum admissible value
            level_to_add1 = icumulative_probas[self.level_spent1 + icumulative_probas <= clevel / 2][0]
        else:
            bk_sup = np.inf
            level_to_add1 = 0
        cumulative_probas = np.hstack([[0], np.cumsum(probas[idx])[:-1]])
        admissible_values_inf = values[
            self.level_spent2 + cumulative_probas <= clevel / 2 ]
        
        if len(admissible_values_inf)>0:
            bk_inf = admissible_values_inf[-1]  # the maximum admissible value
            level_to_add2 = cumulative_probas[self.level_spent2 + cumulative_probas <= clevel / 2][-1]
        else:
            bk_inf = -np.inf
            level_to_add2 = 0
        assert bk_inf <= bk_sup
            
        T = np.sum(Z*(-1)**X)
            
        self.test_stats.append(T)
        
        p_value = 2*min(self.level_spent1 + icumulative_probas[values >= T-1e-12][0] ,
                        self.level_spent2 + cumulative_probas[values <= T+1e-12][-1])

            
        self.level_spent1 += level_to_add1
        self.level_spent2 += level_to_add2

        self.boundary.append((bk_inf, bk_sup))

        self._writer.add_scalar("inf_bound", bk_inf, k)

        self._writer.add_scalar("Stat_val", T, k)

        self._writer.add_scalar("sup_bound", bk_sup, k)

        logger.info(' value of T: '+str(T)+' and boundary: ['+str(bk_inf)+ ','+str( bk_sup)+']')

        if (T > bk_sup) or (T < bk_inf):
            decision = "reject"
        elif k == self.K - 1:
            decision = "accept"
        else:
            decision = "continue"

        return decision, T, (bk_inf, bk_sup), p_value

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
       
        seeders = self.seeder.spawn(2*self.K)
        for k in range(self.K):
            
            m1 = AgentManager(agent_class1, **kwargs1, seed=seeders[2*k])
            m2 = AgentManager(agent_class2, **kwargs2, seed=seeders[2*k+1])

            m1.fit()
            m2.fit()

            Z = np.hstack([Z, self._get_evals(m1), self._get_evals(m2)])
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])


            self.decision, T, (bk_inf, bk_sup), p_val = self.partial_compare(Z, X, k)

            
            if self.decision == "reject":
                logger.info("Reject the null after " + str(k + 1) + " groups")
                if T <= bk_inf:
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

