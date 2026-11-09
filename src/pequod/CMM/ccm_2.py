import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from mpl_toolkits.mplot3d import Axes3D


def inside_domain1(x, y, cx=0.5, cy=0.5, r=0.3):
    """
    Example "inside" function that returns True if (x,y) is
    inside a circle of radius r centered at (cx, cy).
    Replace this with your own geometry routine for arbitrary
    irregular domains.
    """
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

def inside_domain2(x, y):

    # return x <= 0.8 or y >= 0.2
    # return x - y <= 0.8
    c = 0.5
    return (x >= np.pi + c or x <= np.pi - c) or (y <= np.pi - c or y >= np.pi + c)

def inside_domain3(x, y):

    return (x >= 0.3 and y <= 0.7) or (x <= 0.7 and y >= 0.3)

def inside_domain4(x, y):

    return (x <= 0.4 or x >= 0.6) or y <= 0.5

def build_grid_and_masks(a, b, Nx, Ny):
    """
    FV-style grid: unknowns at cell centers.
    Domain is [0,a]x[0,b] with boundary faces at x=0,a and y=0,b,
    which are half a cell away from the outermost centres.
    """
    dx = a / Nx
    dy = b / Ny

    x_vals = (np.arange(Nx) + 0.5) * dx
    y_vals = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    in_mask = np.zeros((Nx, Ny), dtype=bool)
    for i in range(Nx):
        for j in range(Ny):
            in_mask[i, j] = inside_domain2(X[i, j], Y[i, j])

    boundary_mask = np.zeros_like(in_mask)
    for i in range(Nx):
        for j in range(Ny):
            if in_mask[i, j]:
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                for (ii, jj) in neighbors:
                    if ii < 0 or ii >= Nx or jj < 0 or jj >= Ny:
                        boundary_mask[i, j] = True
                        break
                    if not in_mask[ii, jj]:
                        boundary_mask[i, j] = True
                        break

    return X, Y, in_mask, boundary_mask


def build_poisson_matrix_and_rhs(X, Y, in_mask, boundary_mask, rho, outer_bc_func=None):
    """
    FV-style 5-point operator on cell centers for -∇²u = rho (equivalently ∇²u = -rho).

    Unknowns at centers. For neighbors outside the *rectangular* box, we impose Dirichlet
    on the boundary face located half a cell away from the outermost centers.

    For neighbors outside the *irregular* domain (in_mask==False), we do NOT impose anything
    here (that's handled by CMM / irregular boundary treatment). Those connections are simply absent
    from the stencil, leaving those DOFs to be corrected by the capacitance mechanism.
    """
    Nx, Ny = X.shape

    # uniform spacings inferred from centers
    dx = X[1, 0] - X[0, 0] if Nx > 1 else 1.0
    dy = Y[0, 1] - Y[0, 0] if Ny > 1 else 1.0

    # boundary face distances (half-cell)
    dx_b = 0.5 * dx
    dy_b = 0.5 * dy

    if outer_bc_func is None:
        def outer_bc_func(xf, yf):  # homogeneous by default
            return 0.0

    # collect unknowns (cells in_mask==True)
    interior_points = [(i, j) for i in range(Nx) for j in range(Ny) if in_mask[i, j]]
    N_in = len(interior_points)

    idx_map = {p: k for k, p in enumerate(interior_points)}

    A = lil_matrix((N_in, N_in), dtype=np.float64)
    b = np.zeros(N_in, dtype=np.float64)

    for row, (i, j) in enumerate(interior_points):
        # Start with RHS for ∇²u = -rho
        b[row] = -rho[i, j]

        diag = 0.0

        # --- x-direction neighbors ---
        # East (i+1)
        if i + 1 < Nx:
            if (i + 1, j) in idx_map:
                A[row, idx_map[(i + 1, j)]] = 1.0 / dx**2
                diag += -1.0 / dx**2
            else:
                # neighbor is inside rectangle but outside irregular domain -> no coupling
                # (CMM handles irregular boundary)
                pass
        else:
            # outside rectangle: Dirichlet at x = a face, half-cell away
            uB = outer_bc_func(X[i, j] + dx_b, Y[i, j])
            coeff = 1.0 / (dx * dx_b)   # = 2/dx^2 for uniform grid
            diag += -coeff
            b[row] += coeff * uB

        # West (i-1)
        if i - 1 >= 0:
            if (i - 1, j) in idx_map:
                A[row, idx_map[(i - 1, j)]] = 1.0 / dx**2
                diag += -1.0 / dx**2
            else:
                pass
        else:
            uB = outer_bc_func(X[i, j] - dx_b, Y[i, j])
            coeff = 1.0 / (dx * dx_b)
            diag += -coeff
            b[row] += coeff * uB

        # --- y-direction neighbors ---
        # North (j+1)
        if j + 1 < Ny:
            if (i, j + 1) in idx_map:
                A[row, idx_map[(i, j + 1)]] = 1.0 / dy**2
                diag += -1.0 / dy**2
            else:
                pass
        else:
            uB = outer_bc_func(X[i, j], Y[i, j] + dy_b)
            coeff = 1.0 / (dy * dy_b)   # = 2/dy^2
            diag += -coeff
            b[row] += coeff * uB

        # South (j-1)
        if j - 1 >= 0:
            if (i, j - 1) in idx_map:
                A[row, idx_map[(i, j - 1)]] = 1.0 / dy**2
                diag += -1.0 / dy**2
            else:
                pass
        else:
            uB = outer_bc_func(X[i, j], Y[i, j] - dy_b)
            coeff = 1.0 / (dy * dy_b)
            diag += -coeff
            b[row] += coeff * uB

        A[row, row] = diag

    return A, b, idx_map, interior_points


def impose_dirichlet(A, b, idx_map, boundary_nodes, boundary_values):
    """
    Modify the matrix A and vector b to impose Dirichlet boundary
    conditions at the specified boundary_nodes with the given
    boundary_values.  boundary_nodes is a list of (i,j), and
    boundary_values is a dict {(i,j): value, ...}.

    We do this by zeroing out the row, setting A[row,row]=1, and b[row]=the BC.
    """
    for (i, j) in boundary_nodes:
        val = boundary_values[(i, j)]
        row = idx_map[(i, j)]
        # Zero out the row
        A[row, :] = 0.0
        # Put a 1 on the diagonal
        A[row, row] = 1.0
        # RHS
        b[row] = val


def solve_poisson_partial(X, Y, in_mask, boundary_mask, rho):
    """
    FV-style partial solve on the rectangular box with Dirichlet BCs at faces
    (half-cell away). Here we use homogeneous outer BCs by default.
    """
    A, b, idx_map, interior_points = build_poisson_matrix_and_rhs(
        X, Y, in_mask, boundary_mask, rho, outer_bc_func=None
    )

    A_csc = csc_matrix(A)
    sol_vec = spsolve(A_csc, b)

    U1 = np.zeros_like(X, dtype=np.float64)
    for row, (i, j) in enumerate(interior_points):
        U1[i, j] = sol_vec[row]
    return U1


def solve_laplace_with_bc(
        X, Y, in_mask, boundary_mask,
        specified_boundary_nodes, specified_boundary_values
):
    """
    FV-style Laplace solve with outer Dirichlet at faces (homogeneous by default),
    plus pointwise Dirichlet constraints on specified_boundary_nodes (CMM basis).
    """
    rho_zero = np.zeros_like(X)

    A, b, idx_map, interior_points = build_poisson_matrix_and_rhs(
        X, Y, in_mask, boundary_mask, rho_zero, outer_bc_func=None
    )

    # Add specified internal boundary conditions (only those nodes)
    impose_dirichlet(A, b, idx_map, specified_boundary_nodes, specified_boundary_values)

    A_csc = csc_matrix(A)
    sol_vec = spsolve(A_csc, b)

    U = np.zeros_like(X, dtype=np.float64)
    for row, (ii, jj) in enumerate(interior_points):
        U[ii, jj] = sol_vec[row]
    return U


def build_capacitance_matrix(X, Y, in_mask, boundary_mask, boundary_nodes):
    """
    For each boundary node i in boundary_nodes, solve Laplace's equation
    with BC=1 at node i and BC=0 at all other boundary nodes (and 0 on the
    outer rectangle). Then read off the solution at each boundary node j,
    which forms the matrix C_{j,i}.
    """
    N_b = len(boundary_nodes)
    C = np.zeros((N_b, N_b))

    # For each "column" i in the capacitance matrix:
    for i_idx, node_i in enumerate(boundary_nodes):
        # BC: node_i -> 1, all other boundary nodes -> 0
        bc_values = {node_i: 1.0}
        U_i = solve_laplace_with_bc(X, Y, in_mask, boundary_mask, [node_i], bc_values)

        # Now read off the potential at each boundary node j
        for j_idx, node_j in enumerate(boundary_nodes):
            C[j_idx, i_idx] = U_i[node_j[0], node_j[1]]

    return C


def solve_capacitance_system(C, partial_sol, boundary_nodes, g_func):
    """
    Solve for Q in
       partial_sol|_{node_j} + sum_i C_{j,i} Q_i = g(node_j)
    Then return the array Q.
    """
    # Build right-hand side
    N_b = len(boundary_nodes)
    rhs = np.zeros(N_b)
    for j_idx, (i_j, j_j) in enumerate(boundary_nodes):
        # desired BC
        g_val = g_func(i_j, j_j)  # or g_func in continuous coords if you prefer
        # partial solution at that node
        p_val = partial_sol[i_j, j_j]
        rhs[j_idx] = g_val - p_val

    # Solve C Q = rhs
    Q = np.linalg.solve(C, rhs)
    return Q


def combine_solution(U1, G_list, Q):
    """
    Combine partial solution U1 with the linear combination of G_i.
    G_list is a list of solutions G_i, Q is array of charges Q_i.
    """
    U = U1.copy()
    for i in range(len(Q)):
        U += Q[i] * G_list[i]
    return U

# Define the desired Dirichlet condition on the internal boundary
# For demonstration, let's set it to a constant 0 on the internal boundary.
# (One could choose something else, e.g. g=1, or some function.)
def g_func(i, j):
    # i,j are indices, but we can do something with X[i,j], Y[i,j] if we want
    return 0.0

# ----------------------------
# Example main driver
# ----------------------------
if __name__ == "__main__":
    # Rectangle bounding box
    a, b = 1.0, 1.0
    Nx, Ny = 64, 64  # grid resolution

    # Build grid and domain mask
    X, Y, in_mask, boundary_mask = build_grid_and_masks(a, b, Nx, Ny)


    # Define a source function
    def rho_func(x, y):
        # Example: Gaussian source around center
        return np.exp(-40 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


    rho = rho_func(X, Y)

    # Identify the set of internal boundary nodes
    # (points in the domain that are flagged by boundary_mask)
    boundary_nodes = []
    Nx, Ny = X.shape
    for i in range(Nx):
        for j in range(Ny):
            if boundary_mask[i, j]:
                # Must also be truly inside the domain (in_mask==True)
                boundary_nodes.append((i, j))

    # 1) Compute the partial solution U1 with only the outer boundary = 0, ignoring internal boundary
    U1 = solve_poisson_partial(X, Y, in_mask, boundary_mask, rho)

    # 2) Build the capacitance matrix C_{j,i}
    #    For each boundary node i, we solve Laplace with BC=1 at i, 0 at all other boundary nodes,
    #    and 0 on the outer rectangle boundary. Then record the potential at each boundary node j.
    print("Building capacitance matrix C ... (this may take a bit)")
    C = build_capacitance_matrix(X, Y, in_mask, boundary_mask, boundary_nodes)

    # 3) Solve for Q in C Q = (g - U1|_boundary)
    Q = solve_capacitance_system(C, U1, boundary_nodes, g_func)

    # 4) Construct each G_i solution for final combination
    #    We already computed them in build_capacitance_matrix, but let's store them now.
    #    A small optimization: we can store them in that step to avoid re-solving.  For clarity, we do it here again.
    G_list = []
    for i_idx, node_i in enumerate(boundary_nodes):
        bc_values = {node_i: 1.0}
        G_i = solve_laplace_with_bc(X, Y, in_mask, boundary_mask, [node_i], bc_values)
        G_list.append(G_i)

    # pick a representative boundary node
    G = G_list[len(boundary_nodes) // 2]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    G_plot = np.ma.array(G, mask=~in_mask)

    ax.plot_surface(X, Y, G_plot, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_title("Green's function surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("G")
    plt.show()

    # # 5) Combine final solution:  U = U1 + sum_i Q_i G_i
    # U = combine_solution(U1, G_list, Q)
    #
    # # Plot the result
    # fig, ax = plt.subplots(figsize=(6, 5))
    # # Mask out points outside domain
    # U_plot = np.ma.array(U, mask=~in_mask)
    # ctf = ax.contourf(X, Y, U, levels=50, cmap='jet')
    # plt.colorbar(ctf, ax=ax)
    # ax.set_title("Example 2") # Poisson
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # plt.show()