import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


def solve_poisson_green_rect(
        a: float,
        b: float,
        Nx: int,
        Ny: int,
        Mmax: int,
        Nmax: int,
        rho_func,
        g_func=None,
) -> tuple:
    """
    Solve Poisson's equation on a rectangular domain [0,a] x [0,b]:

         ∇² u(x,y) = -ρ(x,y)

    with Dirichlet boundary conditions. If g_func is provided then
    u(x,y) = g(x,y) on the boundary; otherwise homogeneous BC (u=0)
    are assumed.

    When g_func is provided, we decompose the solution as:

         u = v + w,

    where v is a lifting function that satisfies v = g on ∂Ω and is
    computed via solving Laplace's equation (∇² v = 0), and w satisfies
    homogeneous BC and:

         ∇² w = -ρ - ∇² v.

    The homogeneous problem for w is solved via an eigenfunction (Green's
    function) expansion using sine series.

    Parameters:
      - a, b: Domain lengths in x and y.
      - Nx, Ny: Number of grid points in x and y.
      - Mmax, Nmax: Truncation levels for the eigenfunction expansion.
      - rho_func: Callable function, rho_func(x, y), returning ρ(x,y).
      - g_func: Optional callable function, g_func(x, y), returning the
                prescribed boundary value on ∂Ω.

    Returns:
      - X, Y: 2D meshgrid arrays for the spatial domain.
      - u: 2D array with the computed solution.
    """
    # Create spatial grid.
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = a / (Nx - 1)
    dy = b / (Ny - 1)

    # Evaluate the source term ρ(x,y) on the grid.
    rho = rho_func(X, Y)

    # ---------------------------------------------------
    # Step 1: Compute the lifting function v that satisfies
    #         v = g on the boundary.
    #         (If no g_func is given, set v=0.)
    # ---------------------------------------------------
    if g_func is None:
        v = np.zeros_like(X)
    else:
        # Initialize v with boundary values.
        v = np.zeros_like(X)
        # Set boundary values using g_func.
        v[0, :] = g_func(0, y)  # left boundary (x=0)
        v[-1, :] = g_func(a, y)  # right boundary (x=a)
        v[:, 0] = g_func(x, 0)  # bottom boundary (y=0)
        v[:, -1] = g_func(x, b)  # top boundary (y=b)

        # Solve Laplace's equation ∇² v = 0 in the interior
        # using a 5-point finite difference scheme.
        # The unknowns are v[i,j] for i=1..Nx-2, j=1..Ny-2.
        N_in = (Nx - 2) * (Ny - 2)
        A = lil_matrix((N_in, N_in))
        b_vec = np.zeros(N_in)

        def idx(i, j):
            # Map 2D interior indices (i,j) to 1D index.
            return (i - 1) * (Ny - 2) + (j - 1)

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                k = idx(i, j)
                # Coefficients for the 5-point stencil:
                A[k, k] = -2.0 / dx ** 2 - 2.0 / dy ** 2
                # Left neighbor:
                if i - 1 >= 1:
                    A[k, idx(i - 1, j)] = 1.0 / dx ** 2
                else:
                    # i-1 is boundary; add its value to b.
                    b_vec[k] -= v[0, j] / dx ** 2
                # Right neighbor:
                if i + 1 <= Nx - 2:
                    A[k, idx(i + 1, j)] = 1.0 / dx ** 2
                else:
                    b_vec[k] -= v[-1, j] / dx ** 2
                # Bottom neighbor:
                if j - 1 >= 1:
                    A[k, idx(i, j - 1)] = 1.0 / dy ** 2
                else:
                    b_vec[k] -= v[i, 0] / dy ** 2
                # Top neighbor:
                if j + 1 <= Ny - 2:
                    A[k, idx(i, j + 1)] = 1.0 / dy ** 2
                else:
                    b_vec[k] -= v[i, -1] / dy ** 2

        A = csc_matrix(A)
        v_interior = spsolve(A, b_vec)
        # Place the computed interior values back into v.
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                k = idx(i, j)
                v[i, j] = v_interior[k]

    # ---------------------------------------------------
    # Step 2: Form the modified source for w:
    #         w satisfies ∇²w = -ρ - ∇²v, with homogeneous BC.
    #         Compute f_mod = -ρ - Lap(v).
    # ---------------------------------------------------
    # Compute Lap(v) using central differences for interior points.
    Lap_v = np.zeros_like(v)
    # Only interior points are needed.
    Lap_v[1:-1, 1:-1] = (
            (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx ** 2 +
            (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy ** 2
    )
    f_mod = -rho - Lap_v  # Right-hand side for w.

    # ---------------------------------------------------
    # Step 3: Solve for w (with homogeneous BC) via eigenfunction expansion.
    #         The eigenfunction expansion for a rectangle is:
    #
    #         w(x,y) = Σₘ₌₁^Mmax Σₙ₌₁^Nmax sin(mπx/a) sin(nπy/b)
    #                  * E[m,n] * I[m,n],
    #
    #         where E[m,n] = 1/[(mπ/a)² + (nπ/b)²] and
    #         I[m,n] = ∫₀ᵃ∫₀ᵇ sin(mπx'/a) sin(nπy'/b) f_mod(x',y') dx' dy'.
    # ---------------------------------------------------
    m_vals = np.arange(1, Mmax + 1)  # shape (Mmax,)
    n_vals = np.arange(1, Nmax + 1)  # shape (Nmax,)

    # Precompute sine eigenfunctions on the grid:
    sin_mx = np.sin(np.outer(m_vals, x) * np.pi / a)  # shape (Mmax, Nx)
    sin_ny = np.sin(np.outer(n_vals, y) * np.pi / b)  # shape (Nmax, Ny)

    # Project f_mod onto the eigenfunctions.
    # I[m,n] = ∬ sin(mπx/a)*sin(nπy/b)* f_mod(x,y) dx dy.
    I = np.einsum('mi,ij,nj->mn', sin_mx, f_mod, sin_ny) * dx * dy

    # Eigenvalue factors.
    m_grid, n_grid = np.meshgrid(m_vals, n_vals, indexing="ij")
    E = 1.0 / (((m_grid * np.pi / a) ** 2) + ((n_grid * np.pi / b) ** 2))

    # Reconstruct w(x,y) on the grid.
    w = np.einsum('mi,mn,nj->ij', sin_mx, E * I, sin_ny)

    # ---------------------------------------------------
    # Final solution: u = v + w.
    # ---------------------------------------------------
    u = v + w
    return X, Y, u


if __name__ == '__main__':
    # Domain parameters.
    a = 1.0
    b = 1.0
    c = 0.3
    d = 0.3

    # Grid resolution.
    Nx, Ny = 11, 11

    # Truncation parameters for the sine series expansion.
    Mmax, Nmax = 50, 50

    x1 = np.linspace(0, a, Nx)
    y1 = np.linspace(0, b, Ny)
    x2 = x1[:int(Nx * c)]
    y2 = y1[:int(Ny * d)]
    x3 = np.intersect1d(x1, x2)
    y3 = np.intersect1d(y1, y2)
    print(x1, y1, x2, y2, x3, y3)

    X3, Y3 = np.meshgrid(x3, y3, indexing="ij")

    # Define the source term ρ(x,y). For example: a Gaussian centered at (0.5, 0.5).
    def rho_func(x, y):
        return np.exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


    # Define inhomogeneous Dirichlet boundary conditions.
    # For illustration, we set a nonzero boundary function.
    def g_func(x, y):
        # Here we choose a simple function.
        # For example, let g(x,y) = sin(πx) + cos(πy) on the boundary.
        return np.sin(np.pi * x) + np.cos(np.pi * y)


    # Uncomment the next line to use homogeneous BC.
    # g_func = None

    # Solve the Poisson equation.
    X, Y, u = solve_poisson_green_rect(a, b, Nx, Ny, Mmax, Nmax, rho_func, g_func=g_func)

    plt.figure(figsize=(6, 5))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(X, Y, 'ko')
    plt.plot(X3[:, -1], Y3[:, -1], 'ro')
    plt.plot(X3[-1, :], Y3[-1, :], 'ro')

    plt.title("Domain ")
    plt.tight_layout()
    plt.show()

    plt.close("all")

    # Plot the computed solution.
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, u, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution $u(x,y)$ of Poisson's Equation")
    plt.tight_layout()
    # plt.show()

