# Solving Heat Equation using Crank-Nicolson Method

The heat equation in one dimension is written as
$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$

The Crank-Nicolson method is an implicit method of solving the partial differential equation. The heat equation in this method is written as
$$ \frac{u^{n+1}_i - u^{n}_i}{\Delta t} = \frac{\alpha}{2} \left(
    \frac{u^{n+1}_{i+1} - 2u^{n+1}_i + u^{n+1}_{i-1}}{\Delta x^2}
    + \frac{u^{n}_{i+1} - 2u^{n}_i + u^{n}_{i-1}}{\Delta x^2}
\right)

Now if we take known terms (at time t[n]) to RHS and unknown terms (at time t[n+1]) to LHS we get
$$T^{n+1}_i - \frac{r}{2}(T^{n+1}_{i+1} - 2T^{n+1}_i + T^{n+1}_{i-1}) = T^{n}_i + \frac{r}{2}(T^{n}_{i+1} - 2T^{n}_i + T^{n}_{i-1})$$

In matrix form we can write this as
$$(I - A) T^{n+1} = (I + A)T^n $$

where $A$ is the matrix for the centered differences for second order differential with Dirichlet boundary conditions. 

The solution is then $T^{n+1} = (I-A)^{-1} (I+A)T^n$