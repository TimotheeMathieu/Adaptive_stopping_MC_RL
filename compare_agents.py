import logging
import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from alt_agent_manager import AgentManager
import os
from scipy.special import binom
import pdb
from sortedcontainers import SortedList

logger = logging.getLogger(__name__)


class AgentComparator:
    """
    TODO: make a documentation
    """

    def __init__(self, n=10, K=5, alpha=0.05, name="PK", n_evaluations=10):
        self.n = n
        self.K = K
        self.alpha = alpha
        self.name = name
        self.n_evaluations = n_evaluations

    def get_spending_fun(self):
        if self.name == "PK":
            return lambda p: self.alpha * np.log(1 + np.exp(1) * p - p)
        elif self.name == "OF":
            return lambda p: 2 - 2 * stats.norm.cdf(
                stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(p)
            )

    def explore_graph(self, k, Rs, boundary):
        sl = SortedList()
        sl.add([0] * (k + 1))
        records = ( sl, [1], [(0,0)]*(k+1)) # rank score, count of number of path yielding rank, nodes
        for j in range(k + 1):
            cs = [(0, 0)]
            for f in range(2 * self.n):  # Stage f of the network
                old_records = (records[0], records[1],  [(0,0)]*len(records[0]))
                records = (SortedList(), [], [])
                csnew = []
                for id_record in range(len(old_records[0])):  # Step 1
                    u = old_records[0][id_record]
                    cu = old_records[1][id_record]
                    c = old_records[2][id_record]
                    children = _get_children(c, 2 * self.n)  # Step 2
                    if children is not None:
                        for child in children:
                            unew = deepcopy(u)  # Step 3
                            for l in range(k + 1 - j):
                                unew[l] = unew[l] + Rs[j + l][f + 2 * self.n * j] * (
                                      child[1] - c[1]
                                )
                            # Test if can be dropped
                            
                            #can_be_dropped = False
                            #if (len(boundary)>0) and (j != k):
                                #if np.any(unew > boundary[-1]):
                                #    can_be_dropped = True
                                # for l in range(k+1-j):
                                #     test = deepcopy(unew[l])
                                    
                                #     rs = Rs[j + l][(f + 2 * self.n * j):(2*self.n * (j+1))]
                                #     idx = np.argsort(rs)
                                #     test +=  np.sum(rs[idx][:(self.n-f-1)])
                                #     for ll in range(1,l):
                                #         rs = Rs[j + l][( 2 * self.n * (j+ll)):(2*self.n * (j+ll+1))]
                                #         idx = np.argsort(rs)
                                #         test += np.sum(rs[idx][:self.n])
                                #     if test > boundary[-1]:
                                #         can_be_dropped = True
                                #     print(unew, test, boundary[-1], records)
                            if len(records[0]) > 0:
                                is_in = unew in records[0]
                            else:
                                is_in = False
                            #if can_be_dropped == False:
                            if is_in:
                                l = records[0].index(unew)
                                
                            if np.any(is_in) and (records[2][l]==child):  # Step 4
                                records[1][l] += cu
                            else:

                                records[0].add(unew)
                                l = records[0].index(unew)
                                records[1].insert(l,cu)
                                records[2].insert(l, child)
                            # else:
                            #     if np.any(is_in):
                            #         l = np.where(is_in)[0][0]
                            #         records[1].pop(l)
                            #         records[0].pop(l)
                            #         rec_nodes.pop(l)
                            if child not in csnew:
                                csnew.append(child)
                            #print(unew, records[0], child, rec_nodes)
                        
                cs = csnew
            if j != k:
                records = self.postprocess_records(
                    records, boundary
                )
            
        return records

    def postprocess_records(self, records, boundary):
        # Pruning for conditional proba
        idxs = np.where(np.array([r[0] < boundary[-1] for r in records[0]]))[0]
        if len(idxs)>0:
            new_list = [ records[0][i] for i in idxs]
            records = (new_list, list(np.array(records[1])[idxs]),list(np.array(records[2])[idxs]) )

        # Remove the first element of all the records.
        sl = SortedList()
        r1 = []
        r2 = []
        for i in range(len(records[0])):
            sl.add(records[0][i][1:])
            l = sl.index(records[0][i][1:])
            r1.insert(l, records[1][i])
            r2.insert(l, records[2][i])
        records = (sl, r1, r2)
        return records

    def compare(self, manager1, manager2):
        """
        manager1 : tuple of agent_class and init_kwargs for the agent.
        manager2 : tuple of agent_class and init_kwargs for the agent.

        Test whether manager1 is better than manager2
        """
        X = np.array([])
        Z = np.array([])

        agent_class1 = manager1[0]
        kwargs1 = manager1[1]
        kwargs1["n_fit"] = self.n

        agent_class2 = manager1[0]
        kwargs2 = manager2[1]
        kwargs2["n_fit"] = self.n

        spending_fun = self.get_spending_fun()

        boundary = []
        level_spent = 0
        T = 0
        for k in range(self.K):
            m1 = AgentManager(agent_class1, **kwargs1)
            m2 = AgentManager(agent_class2, **kwargs2)

            # m1.fit()
            # m2.fit()

            # Z1 = self._get_rewards(m1)
            # Z2 = self._get_rewards(m2)
            Z1 = np.random.normal(size=self.n)
            Z2 = 3+np.random.normal(size=self.n)

            Z = np.hstack([Z, Z1, Z2])
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])

            Rs = self._get_ranks(Z, k)

            clevel = spending_fun((k + 1) / self.K)

            records = self.explore_graph(k, Rs, boundary)
            ranks = np.array(records[0]).ravel()
            idx = np.argsort(ranks)
            probas = np.array(records[1]) / binom(2 * self.n, self.n) ** (k + 1)
            values = np.array(ranks)[idx]

            cumulative_probas = np.sum(probas)-np.cumsum(probas[idx])
            admissible_values = values[level_spent +cumulative_probas <= clevel ]
            bk = np.min(admissible_values)
            boundary.append(bk)

            
            T = np.sum(Rs[-1] * X)
            level_spent += cumulative_probas[level_spent + cumulative_probas <= clevel][0]
            #breakpoint()

            if T > bk:
                self.decision = "reject"
            else:
                self.decision = "accept"

            if self.decision == "reject":
                logger.info("Reject the null after " + str(k + 1) + " groups")
                logger.info(m1.agent_name + " is better than " + m2.agent_name)
                break
            else:
                logger.info("Did not reject on interim " + str(k + 1))
        if self.decision == "accept":
            logger.info(
                "Did not reject the null hypothesis: either K, n are too small or the agents perform similarly"
            )
        self.rewards_1 = Z[X == 0]
        self.rewards_2 = Z[X == 1]

    def _get_ranks(self, Z, k):
        Rs = []
        for j in range(k + 1):
            Rs.append(np.argsort(Z[: (2 * self.n * (j + 1))])+1)
        return Rs

    def _get_rewards(self, manager):
        """
        Can be overwritten for alternative evaluation function.
        """
        eval_values = []
        for idx in range(self.n):
            logger.info("Evaluating agent " + str(idx))
            eval_values.append(
                np.mean(
                    manager.eval_agents(self.n_evaluations, agent_id=idx, verbose=False)
                )
            )
        return eval_values


def _get_children(c, nmax):
    """
    c is a couple (total size, size of assigned to 1)
    nmax is int, maximum size
    """
    if c[0] == nmax:
        return None
    if c[1] == nmax // 2:
        return [(c[0] + 1, c[1])]
    elif c[0] - c[1] == nmax / 2:
        return [(c[0] + 1, c[1] + 1)]
    elif c[0] < nmax:
        return [(c[0] + 1, c[1]), (c[0] + 1, c[1] + 1)]
    else:
        return None