# Crank–Nicolson Method for the 1D Heat Equation

This document presents the derivation and implementation of the **Crank–Nicolson (CN) method** for solving the one-dimensional heat equation using finite differences.

---

## 1. Problem Statement

Consider a general time-dependent partial differential equation (PDE):

$$
\frac{\partial u}{\partial t} = F(u, x, t, u', u'')
$$

### Example: 1D Heat Equation

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},
\quad x \in (0, L), \quad t \in (0, T)
$$

---

## 2. Discretization

### Spatial Grid

Discretize the spatial domain $(0, L)$ into $n_x$ intervals:

$$
\Delta x = \frac{L}{n_x-1}
$$

### Temporal Grid

Discretize the time domain $(0, T)$ into $n_t$ steps:

$$
\Delta t = \frac{T}{n_t}
$$

### Notation

$$
u(x_i, t_n) \equiv u_i^n,
\quad
x_i = i\Delta x,
\quad
t_n = n\Delta t
$$

---

## 3. Crank–Nicolson Time Discretization

The Crank–Nicolson method is obtained by applying:

- Forward Euler in time  
- Centered differences in space  
- Averaging the spatial operator between time levels $n$ and $n+1$

$$
\frac{u_i^{n+1} - u_i^n}{\Delta t}=\frac{1}{2} \left( F_i^{n+1} + F_i^n \right)
$$

For the heat equation:

$$
F_i^n = \alpha \frac{u_{i-1}^n - 2u_i^n + u_{i+1}^n}{\Delta x^2}
$$

---

## 4. Fully Discrete CN Scheme

Substituting the spatial discretization:

$$
\frac{u_i^{n+1} - u_i^n}{\Delta t}=\frac{\alpha}{2}
\left[
\frac{u_{i-1}^{n+1} - 2u_i^{n+1} + u_{i+1}^{n+1}}{\Delta x^2}+
\frac{u_{i-1}^{n} - 2u_i^{n} + u_{i+1}^{n}}{\Delta x^2}
\right]
$$

Define the dimensionless parameter:

$$
r = \frac{\alpha \Delta t}{\Delta x^2}
$$

Rewriting:

$$
u_i^{n+1}-\frac{r}{2}
\left(
u_{i-1}^{n+1} - 2u_i^{n+1} + u_{i+1}^{n+1}
\right)=
u_i^n+\frac{r}{2}
\left(
u_{i-1}^{n} - 2u_i^{n} + u_{i+1}^{n}
\right)
$$

This is an **implicit system** for $u^{n+1}$.

---

## 5. Matrix Form (Interior Points)

Let $u^n$ be the vector of interior grid values.

Define matrix $A$ as:

$$
A = \frac{r}{2}
\begin{bmatrix}
-2 & 1 & 0 & \cdots \\
1 & -2 & 1 & \cdots \\
0 & 1 & -2 & \cdots \\
\vdots & & & \ddots
\end{bmatrix}
$$

Then the CN scheme becomes:

$$
(I - A) u^{n+1} = (I + A) u^n
$$

Solving:

$$
u^{n+1} = (I - A)^{-1} (I + A) u^n
$$

---

## 6. Boundary Conditions

### Dirichlet Boundary Conditions

$$
u(0, t) = g_0(t),
\quad
u(L, t) = g_L(t)
$$

At time level $t_n$:

$$
u_0^n = g_0^n,
\quad
u_{n_x}^n = g_L^n
$$

Boundary points are handled separately and contribute to the right-hand side.

---

## 7. Modified System with Boundary Terms

After incorporating boundary conditions, the system becomes:

$$
(I - A) u^{n+1} = (I + A) u^n + b
$$

where vector $b$ contains boundary contributions such as:

$$
b_1 = \frac{r}{2} (g_0^{n+1} + g_0^n),
\quad
b_{n_x-1} = \frac{r}{2} (g_L^{n+1} + g_L^n)
$$

Final update formula:

$$
u^{n+1}=(I - A)^{-1} (I + A) u^n+(I - A)^{-1} b
$$