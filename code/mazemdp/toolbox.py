from random import randrange

import numpy as np

# ============ action constants ================
N = 0
S = 1
E = 2
W = 3
NOOP = 4


def discreteProb(p):
    """
    param: p a probability distribution
    Draw a random number using probability table p (column vector)
    Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1
    and the components p(j) are nonnegative.
    To generate a random sample of size m from this distribution,
    imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n).
    Generate a uniform number rand, if this number falls in the jth interval given the discrete distribution,
    #return the value j. Repeat m times.
    """
    r = np.random.random()
    cumprob = np.hstack((np.zeros(1), p.cumsum()))
    sample = -1
    for j in range(p.size):
        if (r > cumprob[j]) & (r <= cumprob[j + 1]):
            sample = j
            break
    return sample
