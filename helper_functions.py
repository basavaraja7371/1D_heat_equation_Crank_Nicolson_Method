import numpy as np
from scipy.sparse import diags


def d2_dirichlet(nx, dx):
    """
    Constructs the centered second-order derivative matrix with Dirichlet
    Boundary conditions.

    Args:
        nx (integer): number of grid points
        dx (float): grid spacing

    Returns:
        d2mat (numpy.ndarry): matrix to compute the centered second order
        derivative with Dirichlet boundary conditions.
    """
    # diagonal elements
    diagonals = [[1.0], [-2.0], [1.0]]

    offsets = [-1, 0, 1]

    d2mat = diags(
        diagonals=diagonals, offsets=offsets, shape=(nx - 2, nx - 2)
    ).toarray()

    return d2mat / dx**2
