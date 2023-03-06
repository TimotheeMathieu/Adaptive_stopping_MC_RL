
import pytest

import sys
sys.path.insert(0, "../")

from joblib import Parallel, delayed
from compare_agents import MultipleAgentsComparator
import numpy as np
import time
    
B = 5000
alpha = 0.05

def test_runtime():
    idxs = []
    comparator = MultipleAgentsComparator(5, 5,B,  alpha=alpha, seed=42, beta = 0.01, joblib_backend = "multiprocessing")
    agents_evals = [np.hstack([np.random.normal(size=5) for k in range(5)]) for n_agents in range(3)]
    decisions = comparator.compare_scalars(agents_evals)


@pytest.mark.parametrize("K,n", [(3, 5), (1, 15)])
def test_type1(K,n):
    idxs = []
    for M in range(100):
        comparator = MultipleAgentsComparator(n, K,B,  alpha=alpha, seed=M, beta = 0.01, joblib_backend = "multiprocessing")
        agents_evals = [np.hstack([np.random.normal(size=n) for k in range(K)]) for n_agents in range(3)]
        decisions = comparator.compare_scalars(agents_evals)
        idxs.append(("smaller" in decisions.values()) or ("larger" in decisions.values()))
    assert np.mean(idxs) < 2*alpha, "type 1 error seems to be too large"
        
if __name__ == "__main__":
    K = 3
    n = 5
    a = time.time()
    comparator = MultipleAgentsComparator(n, K,B,  alpha=alpha, seed=42, beta = 0.01, joblib_backend = "multiprocessing")
    agents_evals = [np.hstack([np.random.normal(size=n) for k in range(K)]) for n_agents in range(3)]
    decisions = comparator.compare_scalars(agents_evals)
    print("Time is ",time.time()-a)
