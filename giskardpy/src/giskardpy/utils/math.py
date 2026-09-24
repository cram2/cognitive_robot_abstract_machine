import numpy as np
import scipy.sparse as sp


def fast_sparse_diagonal(diagonal) -> sp.csc_matrix:
    """
    Faster than scipy.sparse.diags.
    """
    n = len(diagonal)
    return sp.csc_matrix((diagonal, np.arange(n), np.arange(n + 1)), shape=(n, n))
