import numpy as np

def w(z,x):
    R = np.argsort(z, axis=None)
    return R.dot(X)

rng = np.random.RandomState(42)

n = 3
K = 3

Z = np.array([])
X = np.array([])

# Goal : for each value of $W$, count the number of permutations of groups
# can give value $W$

values = [(0,0)]

for i in range(K):
    Z = np.hstack([Z, rng.normal(size=n)])
    X = np.hstack([X, np.zeros(n)])
    
    Z = np.hstack([Z, 2+rng.normal(size=n)])
    X = np.hstack([X, np.ones(n)])

    for f in range(n):
        pass


    W = w(Z, X[rng.permutation(len(X))])
    print(W)

    break



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

# Going through the tree.

records = {(0,0)}
cs = {(0,0)}
print(cs)
# for f in range(6):
#     csnew = set()
#     for c in cs:
#         children = get_children(c,6)
#         if children is not None:
#             for child in children :
#                 csnew.add(child)
#     cs = csnew
#     print(cs)
records = ([0], [1])
for f in range(2*n):
    for id_record in range(len(records[0])):
        u = records[0][id_record]
        cu = records[1][id_record]
        for c in cs:
            children = get_children(c,2*n)
            if children is not None:
                for child in children :
                    u=u + Z[f] * (child[1] - c[1])
                    if u in records[0]:
                        j = records[0].index(u)
                        records[1][j] += cu
                    else:
                        records[0].append(u)
                        records[1].append(cu)
                
print(records)

