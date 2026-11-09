import numpy as np


def rectangle_frame(A, B, C, D, tol=1e-12):
    """
    Build local rectangle coordinates from corners A, B, C, D.
    Assumes A-B-C-D are ordered around the rectangle.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    a1 = B - A
    a2 = D - A

    L = np.linalg.norm(a1)
    H = np.linalg.norm(a2)

    e1 = a1 / L
    e2 = a2 / H

    if abs(np.dot(e1, e2)) > tol:
        raise ValueError(" A, B, C, D do not appear to form an orthogonal rectangle.")

    return A, e1, e2, L, H


def to_local(points, A, e1, e2):
    """
    Convert physical 2D points to local rectangle coordinates (u, v).
    """
    points = np.asarray(points, dtype=np.float64)
    r = points - A
    u = r @ e1
    v = r @ e2
    return np.column_stack([u, v])


def periodic_green_matrix(points_uv, L, H, nx, ny):
    """
    Build G_p(x_i, x_j) using FFT-consistent Fourier modes:

        G_p(r) = 1/(LH) sum_{(m,n)!=0}
                 exp(2 pi i (m u/L + n v/H))
                 / ((2 pi m/L)^2 + (2 pi n/H)^2)

    This is the Green's function for -Delta G = delta - 1/(LH).

    The sum is truncated to the modes represented by an (ny, nx) FFT grid.
    """
    pts = np.asarray(points_uv, dtype=np.float64)
    u = pts[:, 0]
    v = pts[:, 1]
    N = len(pts)

    # Pairwise periodic coordinate differences
    du = u[:, None] - u[None, :]
    dv = v[:, None] - v[None, :]

    # FFT-consistent wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=H / ny)

    G = np.zeros((N, N), dtype=float)

    for kyi in ky:
        for kxi in kx:
            k2 = kxi * kxi + kyi * kyi
            if k2 == 0.0:
                continue

            phase = kxi * du + kyi * dv
            G += np.cos(phase) / k2

    G /= (L * H)
    return G


def estimate_regular_part(L, H, nx, ny, eps=None):
    """
    Estimate R_p on the diagonal:

        R_p = lim_{r -> 0} [G_p(r) + (1/(2pi)) log|r|]

    Since the Fourier sum is truncated, this is a numerical finite-part
    estimate. Use eps comparable to a small fraction of the grid spacing.
    """
    if eps is None:
        eps = 0.25 * min(L / nx, H / ny)

    pts = np.array([[0.0, 0.0], [eps, 0.0]])
    G2 = periodic_green_matrix(pts, L, H, nx, ny)
    G_eps = G2[0, 1]

    return G_eps + (1.0 / (2.0 * np.pi)) * np.log(eps)


def build_capacitance_matrix(
    A, B, C, D,
    points,
    nx,
    ny,
    weights=None,
    use_panel_diagonal=True,
):
    """
    Build capacitance matrix for irregular boundary/source points.

    Parameters
    ----------
    A, B, C, D : array-like, shape (2,)
        Rectangle corners, ordered counterclockwise.
        Period vectors are B-A and D-A.

    points : array-like, shape (N, 2)
        Irregular physical points where charges are placed and boundary
        conditions are collocated.

    nx, ny : int
        FFT grid size. Periodic identification is i=0=nx and j=0=ny.

    weights : None or array-like, shape (N,)
        Boundary quadrature weights / panel lengths.
        If None, no column weighting is applied.

    use_panel_diagonal : bool
        If True and weights are provided, replaces the diagonal by the
        regularized panel self-interaction.

    Returns
    -------
    Cmat : ndarray, shape (N, N)
        Capacitance matrix.

    info : dict
        Geometry and local-coordinate information.
    """
    A0, e1, e2, L, H = rectangle_frame(A, B, C, D)
    points = np.asarray(points, dtype=np.float64)
    points_uv = to_local(points, A0, e1, e2)

    # Wrap into the periodic rectangle
    points_uv[:, 0] = np.mod(points_uv[:, 0], L)
    points_uv[:, 1] = np.mod(points_uv[:, 1], H)

    G = periodic_green_matrix(points_uv, L, H, nx, ny)

    if weights is None:
        Cmat = G.copy()
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != points.shape[0]:
            raise ValueError(" weights must have one entry per point.")

        # Column weighting: integral approx sum_j G(x_i, x_j) q_j w_j
        Cmat = G * weights[None, :]

        if use_panel_diagonal:
            Rp = estimate_regular_part(L, H, nx, ny)

            for j, h in enumerate(weights):
                # self panel integral:
                # ∫[-h/2,h/2] [-(1/2pi)log|s| + Rp] ds
                Cmat[j, j] = (
                    -(h / (2.0 * np.pi)) * (np.log(h / 2.0) - 1.0)
                    + Rp * h
                )

    info = {
        "A": A0,
        "e1": e1,
        "e2": e2,
        "L": L,
        "H": H,
        "points_uv": points_uv,
    }

    return Cmat, info

def rectangle_boundary_indices(nx, ny):
    """
    Return boundary grid indices (j, i) for rectangle boundary,
    avoiding duplicate corners.

    Grid:
        j = 0,...,ny-1
        i = 0,...,nx-1

    Periodic identification:
        i = 0 == nx
        j = 0 == ny
    """
    inds = []

    # bottom: j = 0, i = 0,...,nx-1
    for i in range(nx):
        inds.append((0, i))

    # right: i = nx-1, j = 1,...,ny-1
    for j in range(1, ny):
        inds.append((j, nx - 1))

    # top: j = ny-1, i = nx-2,...,0
    for i in range(nx - 2, -1, -1):
        inds.append((ny - 1, i))

    # left: i = 0, j = ny-2,...,1
    for j in range(ny - 2, 0, -1):
        inds.append((j, 0))

    return np.array(inds, dtype=int)

def apply_capacitance_correction(grid_points, boundary_points, q, weights,
                                 A, B, C, D, nx, ny):
    """
    Compute u_c(x) = sum_j G_p(x, y_j) q_j w_j
    at arbitrary grid_points.
    """
    A0, e1, e2, L, H = rectangle_frame(A, B, C, D)

    x_uv = to_local(grid_points, A0, e1, e2)
    y_uv = to_local(boundary_points, A0, e1, e2)

    x_uv[:, 0] = np.mod(x_uv[:, 0], L)
    x_uv[:, 1] = np.mod(x_uv[:, 1], H)
    y_uv[:, 0] = np.mod(y_uv[:, 0], L)
    y_uv[:, 1] = np.mod(y_uv[:, 1], H)

    u = x_uv[:, 0]
    v = x_uv[:, 1]
    y_u = y_uv[:, 0]
    y_v = y_uv[:, 1]

    du = u[:, None] - y_u[None, :]
    dv = v[:, None] - y_v[None, :]

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=H / ny)

    uc = np.zeros(len(grid_points), dtype=float)

    charge = q * weights

    for kyi in ky:
        for kxi in kx:
            k2 = kxi * kxi + kyi * kyi
            if k2 == 0.0:
                continue

            phase = kxi * du + kyi * dv
            Gxy_charge = np.cos(phase) @ charge
            uc += Gxy_charge / k2

    uc /= (L * H)
    return uc


def mixed_green_matrix(points_uv, L, H, nx, ny):
    """
    Discrete Green's function for:
        periodic in x
        zero Dirichlet in y

    Grid:
        i = 0,...,nx-1
        j = 1,...,ny

    y=0 and y=H are Dirichlet boundaries, and ny is the number of
    interior y points.
    """
    pts = np.asarray(points_uv, dtype=float)
    u = pts[:, 0]
    v = pts[:, 1]

    dx = L / nx
    dy = H / (ny + 1)

    # Convert physical points to nearest grid indices
    i = np.round(u / dx).astype(int) % nx
    j = np.round(v / dy).astype(int)

    if np.any(j <= 0) or np.any(j > ny):
        raise ValueError("Some points lie outside the interior y-domain.")

    N = len(pts)
    G = np.zeros((N, N), dtype=float)

    m_vals = np.arange(nx)
    n_vals = np.arange(1, ny + 1)

    # discrete eigenvalues
    lam_x = 4.0 * np.sin(np.pi * m_vals / nx)**2 / dx**2
    lam_y = (
        4.0
        * np.sin(np.pi * n_vals / (2.0 * (ny + 1))) ** 2
        / dy**2
    )

    # sine eigenfunctions in y, normalized
    Phi = np.sqrt(2.0 / (ny + 1)) * np.sin(
        np.pi * np.outer(j, n_vals) / (ny + 1)
    )

    for m, lx in zip(m_vals, lam_x):
        phase = 2.0 * np.pi * m * (i[:, None] - i[None, :]) / nx
        cos_phase = np.cos(phase)

        denom = lx + lam_y
        vertical = (Phi / denom) @ Phi.T

        G += cos_phase * vertical

    G /= nx
    return G

def build_capacitance_matrix_mixed_bc(
    A, B, C, D,
    points,
    nx,
    ny,
    weights=None,
):
    """
    Capacitance matrix using Green's function with:
        periodic BC in x
        zero Dirichlet BC in y
    """
    A0, e1, e2, L, H = rectangle_frame(A, B, C, D)

    points = np.asarray(points, dtype=float)
    points_uv = to_local(points, A0, e1, e2)

    # periodic wrap only in x
    points_uv[:, 0] = np.mod(points_uv[:, 0], L)

    G = mixed_green_matrix(points_uv, L, H, nx, ny)

    if weights is None:
        Cmat = G
    else:
        weights = np.asarray(weights, dtype=float)
        Cmat = G * weights[None, :]

    info = {
        "A": A0,
        "e1": e1,
        "e2": e2,
        "L": L,
        "H": H,
        "points_uv": points_uv,
    }

    return Cmat, info

def apply_capacitance_correction_mixed_bc(
    grid_points,
    boundary_points,
    q,
    A, B, C, D,
    nx, ny,
    X_shape,
    weights=None,
):
    A0, e1, e2, L, H = rectangle_frame(A, B, C, D)

    x_uv = to_local(grid_points.reshape(-1, 2), A0, e1, e2)
    y_uv = to_local(boundary_points, A0, e1, e2)

    x_uv[:, 0] = np.mod(x_uv[:, 0], L)
    y_uv[:, 0] = np.mod(y_uv[:, 0], L)

    Gxy = mixed_green_matrix_cross(x_uv, y_uv, L, H, nx, ny)

    charge = q if weights is None else q * weights
    uc = Gxy @ charge

    return uc.reshape(X_shape)

def mixed_green_matrix_cross(x_uv, y_uv, L, H, nx, ny):
    dx = L / nx
    dy = H / (ny + 1)

    ix = np.round(x_uv[:, 0] / dx).astype(int) % nx
    jx = np.round(x_uv[:, 1] / dy).astype(int)

    iy = np.round(y_uv[:, 0] / dx).astype(int) % nx
    jy = np.round(y_uv[:, 1] / dy).astype(int)

    m_vals = np.arange(nx)
    n_vals = np.arange(1, ny + 1)

    lam_x = 4.0 * np.sin(np.pi * m_vals / nx)**2 / dx**2
    lam_y = (
        4.0
        * np.sin(np.pi * n_vals / (2.0 * (ny + 1))) ** 2
        / dy**2
    )

    Phi_x = np.sqrt(2.0 / (ny + 1)) * np.sin(
        np.pi * np.outer(jx, n_vals) / (ny + 1)
    )
    Phi_y = np.sqrt(2.0 / (ny + 1)) * np.sin(
        np.pi * np.outer(jy, n_vals) / (ny + 1)
    )

    G = np.zeros((len(x_uv), len(y_uv)))

    for m, lx in zip(m_vals, lam_x):
        phase = 2.0 * np.pi * m * (ix[:, None] - iy[None, :]) / nx
        vertical = (Phi_x / (lx + lam_y)) @ Phi_y.T
        G += np.cos(phase) * vertical

    G /= nx
    return G
