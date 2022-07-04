import logging
import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
import os

logger = logging.getLogger(__name__)


class AgentComparer:
    """
    TODO: make a documentation
    """

    def __init__(
        self,
        n=10,
        K=5,
        alpha=0.05,
        name="PK",
        eval_tag="episode_rewards",
    ):
        self.n = n
        self.K = K
        self.alpha = alpha
        self.eval_tag = eval_tag
        self.decision = "accept"

    def explore_graph(self,records, R):
        cs = [(0,0)]
        for f in range(2*n):
            for id_record in range(len(records[0])):
                u = records[0][id_record]
                cu = records[1][id_record]
                csnew = []
                for c in cs:
                    children = get_children(c,2*n)
                    if children is not None:
                        for child in children :
                            u=u + R[f+n*i] * (child[1] - c[1])
                            if u in records[0]:
                                j = records[0].index(u)
                                records[1][j] += cu
                            else:
                                records[0].append(u)
                                records[1].append(cu)
                            csnew.append(child)
                cs = csnew
        return records

    def compare(self, manager1, manager2):
        """
        agent1 : tuple of agent_class and init_kwargs for the agent.
        agent2 : tuple of agent_class and init_kwargs for the agent.
        """
        X = np.array([])
        Z = np.array([])
        records = ([0], [1])

        for k in range(self.K):
            m1 = deepcopy(manager1)
            m2 = deepcopy(manager2)

            m1.n_fit = self.n
            m2.n_fit = self.n
            m1.fit()
            m2.fit()

            Z1  = self._get_rewards(m1)
            Z2  = self._get_rewards(m2)
            Z = np.hstack([Z, Z1, Z2])
            X = np.hstack([X, np.zeros(self.n), np.ones(self.n)])

            R = np.argsort(Z, axis=None)

            records = self.explore_graph(records, R)

            idx = np.argsort(records[0])
            probas = np.array(records[1])
            emp_cdf = np.cumsum(probas[idx]/np.sum(probas))
            values = np.array(records[0])[idx]

            q1, q2=  np.max(values[emp_cdf<alpha/2]),np.min(values[emp_cdf>1-alpha/2])
            T = np.sum(R*X)
            if not(q1 < T < q2) :
                self.decision = "reject"
    
            if self.decision == "reject":
                logger.info("Reject the null after " + str(k + 1) + " groups")
                if np.sum(X - Y) > 0:
                    logger.info(m1.agent_name + " is better than " + m2.agent_name)
                else:
                    logger.info(m2.agent_name + " is better than " + m1.agent_name)
                break
            else:
                logger.info("Did not reject on interim " + str(k + 1))
        if k == self.K - 1:
            logger.info(
                "Did not reject the null hypothesis: either K, n are too small or the agents perform similarly"
            )

    def _get_rewards(self, manager):
        writer_data = manager.get_writer_data()
        return [
            np.sum(
                writer_data[idx].loc[writer_data[idx]["tag"] == self.eval_tag, "value"]
            )
            for idx in writer_data
        ]

def _get_children(c, nmax):
    """
    c is a couple (total size, size of assigned to 1)
    nmax is int, maximum size
    """
    if c[0] == nmax:
        return None
    if c[1]== nmax//2:
        return [(c[0]+1, c[1])]
    elif c[0]-c[1]==nmax/2:
        return [(c[0]+1, c[1]+1)]
    elif c[0] < nmax:
        return [(c[0]+1, c[1]), (c[0]+1, c[1]+1)]
    else:
        return None
