import numpy as np
import matplotlib.pyplot as plt


def solve_poisson_green_rect(a: float, b: float, Nx: int, Ny: int,
                             Mmax: int, Nmax: int, rho_func) -> tuple:
    """
    Solve Poisson's equation on a rectangular domain [0,a] x [0,b]
    with homogeneous Dirichlet boundary conditions using a Green's
    function approach via eigenfunction expansion.

    The PDE is:
        ∇² u(x,y) = -ρ(x,y)    for (x,y) in (0,a)×(0,b)
    with u = 0 on the boundary.

    Parameters:
      - a, b: Domain lengths in x and y.
      - Nx, Ny: Number of grid points in x and y directions.
      - Mmax, Nmax: Truncation levels for the eigenfunction expansion.
      - rho_func: Callable function, rho_func(x, y), returning the source term.

    Returns:
      - X, Y: 2D meshgrid arrays for the spatial domain.
      - u: 2D array with the computed solution.
    """
    # Create spatial grid
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = a / (Nx - 1)
    dy = b / (Ny - 1)

    # Evaluate the source term on the grid.
    rho = rho_func(X, Y)

    # Create arrays of mode indices.
    m_vals = np.arange(1, Mmax + 1)  # shape (Mmax,)
    n_vals = np.arange(1, Nmax + 1)  # shape (Nmax,)

    # Precompute the sine eigenfunctions:
    # sin_mx: shape (Mmax, Nx) where [m,i] = sin(mπx_i/a)
    sin_mx = np.sin(np.outer(m_vals, x) * np.pi / a)
    # sin_ny: shape (Nmax, Ny) where [n,j] = sin(nπy_j/b)
    sin_ny = np.sin(np.outer(n_vals, y) * np.pi / b)

    # Compute the projections of the source term onto the eigenfunctions.
    # I[m, n] = ∬ sin(mπx/a)*sin(nπy/b)*ρ(x,y) dx dy.
    # Use einsum to perform the double summation in one call.
    I = np.einsum('mi,ij,nj->mn', sin_mx, rho, sin_ny) * dx * dy

    # Compute the eigenvalue factors E[m,n] = 1/((mπ/a)^2 + (nπ/b)^2)
    m_grid, n_grid = np.meshgrid(m_vals, n_vals, indexing='ij')
    E = 1.0 / (((m_grid * np.pi / a)**2) + ((n_grid * np.pi / b)**2))

    # Reconstruct the solution:
    # u(x,y) = ∑ₘₙ sin(mπx/a)*sin(nπy/b)*E[m,n]*I[m,n].
    # We perform this double sum for all grid points at once.
    u = np.einsum('mi,mn,nj->ij', sin_mx, E * I, sin_ny)

    return X, Y, u


# ----------------------------
# Example usage
# ----------------------------
if __name__ == '__main__':
    # Domain parameters
    a = 1.0
    b = 1.0

    # Grid resolution
    Nx, Ny = 128, 128

    # Truncation parameters for the series expansion.
    # (Increase these for higher accuracy at the cost of runtime.)
    Mmax, Nmax = 50, 50

    # Define the source term (rho) as a function.
    # For example: a Gaussian centered at (0.5, 0.5).
    def rho_func(x, y):
        return np.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))

    # Solve Poisson's equation.
    X, Y, u = solve_poisson_green_rect(a, b, Nx, Ny, Mmax, Nmax, rho_func)

    # Plot the computed solution.
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, u, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Solution $u(x,y)$ of Poisson's Equation")
    plt.tight_layout()
    plt.show()
