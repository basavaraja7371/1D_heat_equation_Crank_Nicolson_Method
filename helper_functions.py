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


def crank_nicolson_matrices(nx, dx, dt, alpha):
    """
    Create matrices involved in Crank-Nicolson method. (I-A) on the LHS, (I+A)
    on the RHS and vector r.

    Args:
        nx (integer): number of grid points
        dx (float): grid spacing
        dt (float): time step length
        alpha (float): thermal conductivity

    Returns:
        RHS (np.ndarray): (I - A) matrix
        LHS (np.ndarray): (I + A) matrix
        r (np.array): alpha * dt / dx**2
    """
    r = dt * alpha / dx**2

    D2 = d2_dirichlet(nx, dx)
    A = (r / 2) * D2

    Ix = np.eye(nx - 2)

    LHS = Ix - A
    RHS = Ix + A

    return LHS, RHS, r


def boundary_vector(r, g0_n, g0_np1, gL_n, gL_np1, nx):
    """
    Create the boundary vector constituting of boundary condition for
    Crank-Nicolson method.

    Args:
        r (np.array): r vector dt * alpha / dx**2
        g0_n (float): Left boundary condtion at time t[n]
        g0_np1 (float): Left boundary condition at time t[n+1]
        gL_n (float): Right boundary condtion at time t[n]
        gL_np1 (float): Right boundary condtion at time t[n+1]
        nx (float): number of grid points

    Returns:
        b (np.array): boundary vector
    """
    b = np.zeros(nx - 2)
    b[0] = (r / 2) * (g0_n + g0_np1)
    b[-1] = (r / 2) * (gL_n + gL_np1)

    return b


def crank_nicolson_step(u_n, LHS, RHS, b):
    """
    One Crank-Nicolson step to calculate u[n+1] from u[n]

    Args:
        u_n (np.array): u[n] at time t[n]
        LHS (np.ndarray): LHS matrix (I - A)
        RHS (np.ndarray): RHS matrix (I + A)
        b (np.array): boundary value vector

    Returns:
        u_np1 (np.array): u[n+1] at time t[n+1]
    """
    rhs = RHS @ u_n + b
    u_np1 = np.linalg.solve(LHS, rhs)
    return u_np1
