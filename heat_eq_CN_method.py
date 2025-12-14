import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


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
        r (np.array): alpha * dt
    """
    r = dt * alpha

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


def solve_heat_CN(f, g0, gL, L, T, nx, nt, alpha):
    """
    Solves the Heat Equation using Crank-Nicolson method.

    Args:
        f (function): initial condition
        g0 (function): left boundary condition
        gL (function): right boundary condition
        L (float): length of the rod
        T (float): total time
        nx (float): number of grid points
        nt (float): number of time steps
        alpha (float): thermal conductivity

    Returns:
        u (np.array): solution to heat equation
    """
    dx = L / (nx - 1)
    dt = T / nt

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)

    # initial condition (interior solution)
    u = np.zeros(nx)
    u[0] = g0(T)  # at x=0
    u[-1] = gL(T)  # at x=L
    u[1:-1] = f(x)[1:-1]  # interior points

    LHS, RHS, r = crank_nicolson_matrices(nx, dx, dt, alpha)

    for n in range(nt):
        # define boundary condition
        b = boundary_vector(r, g0(t[n]), g0(t[n + 1]), gL(t[n]), gL(t[n + 1]), nx)

        u[1:-1] = crank_nicolson_step(u[1:-1], LHS, RHS, b)

    return x, u


# Let's solve heat equation
# The exact solution is


def exact_solution(x, t, alpha):
    return np.exp(-(np.pi**2) * alpha * t) * np.sin(np.pi * x)


# Initial condition
def f(x):
    return np.sin(np.pi * x)


# Left boundary
def g0(t):
    return 0.0


# Right boundary
def gL(t):
    return 0.0


x, u_final = solve_heat_CN(f=f, g0=g0, gL=gL, L=1.0, T=5, nx=50, nt=500, alpha=0.1)


u_exact = exact_solution(x, t=5, alpha=0.1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlabel(r"Length of the rod, $x$", fontsize=18)
ax.set_ylabel(r"Temperature, $T$", fontsize=18)


ax.plot(x, u_final, linewidth=2, label="Crank-Nicolson")
ax.plot(x, u_exact, "r^", markersize=7, alpha=0.7, label="Exact")

ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("heat_equation_CN.jpg")
plt.show()
