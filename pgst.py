import numpy as np

def get_children(c, nmax):
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

rng = np.random.RandomState(42)

n = 3
K = 3
alpha = 0.05

Z = np.array([])
X = np.array([])

# Goal : for each value of $W$, count the number of permutations of groups
# can give value $W$

values = [(0,0)]
records = ([0], [1])

for i in range(K):
    Z = np.hstack([Z, rng.normal(size=n)])
    X = np.hstack([X, np.zeros(n)])
    
    Z = np.hstack([Z, rng.normal(size=n)])
    X = np.hstack([X, np.ones(n)])

    R = np.argsort(Z, axis=None)
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
                
    #print(records)
    idx = np.argsort(records[0])
    probas = np.array(records[1])
    emp_cdf = np.cumsum(probas[idx]/np.sum(probas))
    values = np.array(records[0])[idx]
    print(np.max(values[emp_cdf<alpha/2]),np.min(values[emp_cdf>1-alpha/2]))
    print(np.sum(R*X))
    print('*'*10)
