import math
from z3 import Or, And


def sort_net(a):
    """
    Batcher's odd-even mergesort described by
    Knuth in The Art of Computer Programming vol. 3
    """
    n = len(a)
    t = math.ceil(math.log2(n))
    p = 2 ** (t - 1)
    while p > 0:
        q = 2 ** (t - 1)
        r = 0
        d = p
        while d > 0:
            for i in range(n - d):
                if i & p == r:
                    a[i], a[i + d] = Or(a[i], a[i + d]), And(a[i], a[i + d])
            d = q - p
            q //= 2
            r = p
        p //= 2
