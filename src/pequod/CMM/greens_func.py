import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Domain and discretization
# ----------------------------
a = 1.0  # Domain: 0 <= x <= a
b = 1.0  # Domain: 0 <= y <= b

# Use a relatively coarse grid (for speed); one may refine as needed.
Nx = 20  # number of grid points in x
Ny = 20  # number of grid points in y

# Create 1D arrays for x and y.
x = np.linspace(0, a, Nx)
y = np.linspace(0, b, Ny)

# We use meshgrid with 'ij' indexing so that u[i,j] corresponds to (x[i], y[j]).
X, Y = np.meshgrid(x, y, indexing='ij')


# ----------------------------
# Source term ρ(x,y)
# ----------------------------
# For example, choose a source localized near the center.
def rho_func(x, y):
    return np.exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


rho = rho_func(X, Y)

# Integration weights (using the rectangular rule)
dx = a / (Nx - 1)
dy = b / (Ny - 1)

# ----------------------------
# Truncation for the series expansion
# ----------------------------
Mmax = 10  # maximum index in x-direction for the series
Nmax = 10  # maximum index in y-direction for the series

# Precompute the eigenfunction sine factors for the x and y grids.
# For m = 1,2,...,Mmax and x grid:
m_vals = np.arange(1, Mmax + 1)  # shape (Mmax,)
sin_mx = np.sin(np.outer(m_vals, x) * np.pi / a)  # shape (Mmax, Nx)

# For n = 1,2,...,Nmax and y grid:
n_vals = np.arange(1, Nmax + 1)  # shape (Nmax,)
sin_ny = np.sin(np.outer(n_vals, y) * np.pi / b)  # shape (Nmax, Ny)

# Precompute the eigenvalue denominator for the Green's function.
# Create 2D arrays for m and n (with m along rows, n along columns)
M_grid, N_grid = np.meshgrid(m_vals, n_vals, indexing='ij')
E = 1.0 / (((M_grid * np.pi / a) ** 2) + ((N_grid * np.pi / b) ** 2))  # shape (Mmax, Nmax)

# ----------------------------
# Compute the solution u(x,y)
# ----------------------------
# The Green's function is written as a double series. Then,
#   u(x,y) = ∬ G(x,y;x',y') ρ(x',y') dx' dy'
# For fixed evaluation point (x, y), the series representation reads:
#
#   G(x,y;x',y') = sum_{m=1}^{Mmax} sum_{n=1}^{Nmax} [ sin(mπx/a)*sin(mπx'/a) *
#                                                      sin(nπy/b)*sin(nπy'/b) ] * E[m,n]
#
# We perform the integration over (x', y') by summing on the grid.
#
# To speed things up we precompute for each (x,y) evaluation point the
# series contribution “vectorized” over the integration grid.
#
# We use two nested loops over the evaluation grid points (which is modest in size).

u = np.zeros((Nx, Ny))

# Loop over evaluation (target) points (x[i], y[j])
for i in range(Nx):
    for j in range(Ny):
        # For the fixed evaluation point (x[i], y[j]), extract the eigenfunction values:
        # A[m] = sin(mπx[i]/a) for m=1,...,Mmax.
        A = sin_mx[:, i]  # shape (Mmax,)
        # C[n] = sin(nπy[j]/b) for n=1,...,Nmax.
        C = sin_ny[:, j]  # shape (Nmax,)

        # Now, for every integration point (x', y'), we need to compute:
        #   G(x[i],y[j]; x',y') = sum_{m,n} A[m]*sin_mx[m, (integration index)]
        #                                  * C[n]*sin_ny[n, (integration index)] * E[m,n].
        #
        # We vectorize the integration over x' and y' as follows:
        #
        #   - For the x-part, form an array temp_x of shape (Mmax, Nx):
        #         temp_x[m, ip] = A[m] * sin_mx[m, ip]
        #   - For the y-part, form an array temp_y of shape (Nmax, Ny):
        #         temp_y[n, jp] = C[n] * sin_ny[n, jp]
        #
        # Then, using tensor contraction, we combine these with E.
        temp_x = A[:, None] * sin_mx  # shape: (Mmax, Nx)
        temp_y = C[:, None] * sin_ny  # shape: (Nmax, Ny)

        # Contract the m-index of temp_x with the m-index of E.
        # temp_x_E will have shape (Nx, Nmax)
        temp_x_E = np.tensordot(temp_x, E, axes=(0, 0))  # sums over m

        # Now, contract over the n-index with temp_y.
        # The dot product gives an array G_all of shape (Nx, Ny), i.e. the Green's function
        # evaluated at all integration points (x',y') for the fixed (x[i], y[j]).
        G_all = np.dot(temp_x_E, temp_y)  # shape: (Nx, Ny)

        # Finally, approximate the integral over x' and y' by a double sum:
        #   u(x[i], y[j]) ≈ Σ_{ip, jp} G_all[ip, jp] * ρ(x[ip], y[jp]) * dx * dy
        u[i, j] = np.sum(G_all * rho * dx * dy)

# ----------------------------
# Plot the computed solution
# ----------------------------
plt.figure(figsize=(6, 5))
contours = plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar(contours)
plt.title("Solution $u(x,y)$ of Poisson's Equation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
