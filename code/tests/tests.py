import pytest

import sys
sys.path.insert(0, "../")

from joblib import Parallel, delayed
from compare_agents import MultipleAgentsComparator
import numpy as np
    
B = 5000
alpha = 0.05

@pytest.mark.parametrize("K,n", [(3, 5), (1, 15)])
def test_type1(K,n):
    idxs = []
    for M in range(100):
        comparator = MultipleAgentsComparator(n, K,B,  alpha=alpha, seed=M, beta = 0.01, joblib_backend = "multiprocessing")
        agents_evals = [np.hstack([np.random.normal(size=n) for k in range(K)]) for n_agents in range(3)]
        decisions = comparator.compare_scalars(agents_evals)
        idxs.append("reject" in decisions)
    assert np.mean(idxs) < 2*alpha, "type 1 error seems to be too large"
        
