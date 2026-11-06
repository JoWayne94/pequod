"""
Doubly periodic Poisson solver using the 5-point Laplacian and FFT.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import erf
from scipy.fft import rfft2, irfft2, rfft, irfft, dst, idst
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from pequod.CMM.ccm_test import *


class PoissonSolverChannel:
    """
    Solve Δu = f on a 2D grid.
    """
    def __init__(self, shape, coords):
        """
        shape : (ny, nx)
            Number of interior unknowns in y, and periodic points in x.
        coords : [west, south, east, north]
            Domain boundaries.
        """
        assert len(coords) == 4

        self.west, self.south, self.east, self.north = coords
        ny, nx = shape
        self.shape = shape

        Lx = self.east - self.west
        Ly = self.north - self.south

        # Periodic x: nx points, endpoint excluded
        self.dx = Lx / nx
        self.x_coords = self.west + self.dx * np.arange(nx)

        # Dirichlet y: ny interior points, boundaries excluded
        self.dy = Ly / (ny + 1)
        self.y_coords = self.south + self.dy * np.arange(1, ny + 1)

        kx = np.arange(nx // 2 + 1)[None, :]
        jy = np.arange(1, ny + 1)[:, None]

        # Periodic 2nd-difference eigenvalues in x
        lam_x = (
            2.0 * np.cos(2.0 * np.pi * kx / nx) - 2.0
        ) / self.dx**2

        # Dirichlet 2nd-difference eigenvalues in y, DST-I
        lam_y = (
            2.0 * np.cos(np.pi * jy / (ny + 1)) - 2.0
        ) / self.dy**2

        lam = lam_x + lam_y

        # No zero mode because Dirichlet y removes nullspace
        self.inv_lam = 1.0 / lam

    def solve(self, f):
        """
        f : ndarray, shape (ny, nx)
            RHS values at interior y points and periodic x points.
        returns : ndarray, shape (ny, nx)
            Solution at interior y points.
        """
        f = np.asarray(f, dtype=np.float64)

        assert f.shape == self.shape

        # FFT in x
        fhat = rfft(f, axis=1)

        # DST-I in y
        fhat = dst(fhat, type=1, axis=0, norm="ortho")

        # Divide by eigenvalues
        uhat = fhat * self.inv_lam

        # Inverse DST-I in y
        u = idst(uhat, type=1, axis=0, norm="ortho")

        # Inverse FFT in x
        u = irfft(u, n=self.shape[1], axis=1)

        return u.real


class PoissonSolverBasin:
    """Solve ``Delta u = f`` with zero Dirichlet data on all four walls."""

    def __init__(self, shape, coords):
        if len(coords) != 4:
            raise ValueError(
                "coords must be [west, south, east, north]."
            )

        self.west, self.south, self.east, self.north = coords
        ny, nx = shape
        if nx < 1 or ny < 1:
            raise ValueError("Basin Poisson shape must be positive.")
        self.shape = (ny, nx)

        self.dx = (self.east - self.west) / (nx + 1)
        self.dy = (self.north - self.south) / (ny + 1)
        self.x_coords = self.west + self.dx * np.arange(1, nx + 1)
        self.y_coords = self.south + self.dy * np.arange(1, ny + 1)

        ix = np.arange(1, nx + 1)[None, :]
        iy = np.arange(1, ny + 1)[:, None]
        lam_x = (
            2.0 * np.cos(np.pi * ix / (nx + 1)) - 2.0
        ) / self.dx**2
        lam_y = (
            2.0 * np.cos(np.pi * iy / (ny + 1)) - 2.0
        ) / self.dy**2
        self.inv_lam = 1.0 / (lam_x + lam_y)

    def solve(self, f):
        f = np.asarray(f, dtype=np.float64)
        if f.shape != self.shape:
            raise ValueError(
                f"Expected basin RHS shape {self.shape}, got {f.shape}."
            )
        fhat = dst(f, type=1, axis=0, norm="ortho")
        fhat = dst(fhat, type=1, axis=1, norm="ortho")
        uhat = fhat * self.inv_lam
        u = idst(uhat, type=1, axis=1, norm="ortho")
        return idst(u, type=1, axis=0, norm="ortho")


class PoissonSolverBasinMasked:
    """Five-point basin Poisson solve with grid-aligned solid points.

    ``fluid_mask`` has the full solver grid shape. Row zero and column zero
    are the physical south and west Dirichlet walls; the north and east
    walls lie one grid spacing beyond the final stored row and column.
    False mask entries are zero-Dirichlet solid points.
    """

    def __init__(self, fluid_mask, coords):
        if len(coords) != 4:
            raise ValueError(
                "coords must be [west, south, east, north]."
            )
        self.west, self.south, self.east, self.north = coords
        self.fluid_mask = np.asarray(fluid_mask, dtype=bool)
        if self.fluid_mask.ndim != 2:
            raise ValueError("fluid_mask must be a two-dimensional array.")

        ny, nx = self.fluid_mask.shape
        self.shape = (ny, nx)
        self.dx = (self.east - self.west) / nx
        self.dy = (self.north - self.south) / ny

        unknown = self.fluid_mask.copy()
        unknown[0, :] = False
        unknown[:, 0] = False
        self.unknown_mask = unknown
        self._indices = -np.ones(self.shape, dtype=int)
        self._indices[unknown] = np.arange(np.count_nonzero(unknown))
        if not np.any(unknown):
            raise ValueError("The basin has no fluid interior points.")

        rows = []
        cols = []
        values = []
        diagonal = -2.0 / self.dx**2 - 2.0 / self.dy**2
        neighbours = (
            (-1, 0, 1.0 / self.dy**2),
            (1, 0, 1.0 / self.dy**2),
            (0, -1, 1.0 / self.dx**2),
            (0, 1, 1.0 / self.dx**2),
        )

        for j, i in np.argwhere(unknown):
            row = self._indices[j, i]
            rows.append(row)
            cols.append(row)
            values.append(diagonal)
            for dj, di, coefficient in neighbours:
                jj, ii = j + dj, i + di
                if (
                    0 <= jj < ny
                    and 0 <= ii < nx
                    and unknown[jj, ii]
                ):
                    rows.append(row)
                    cols.append(self._indices[jj, ii])
                    values.append(coefficient)

        matrix = coo_matrix(
            (values, (rows, cols)),
            shape=(np.count_nonzero(unknown),) * 2,
        ).tocsc()
        self._factor = splu(matrix)

    def solve(self, f):
        f = np.asarray(f, dtype=np.float64)
        if f.shape != self.shape:
            raise ValueError(
                f"Expected masked basin RHS shape {self.shape}, got {f.shape}."
            )
        result = np.zeros(self.shape, dtype=np.float64)
        result[self.unknown_mask] = self._factor.solve(
            f[self.unknown_mask]
        )
        return result


class PoissonSolver:
    """
    Solve Delta u = f on a 2D periodic grid using the 5-point stencil.
    """
    def __init__(self, shape, coords):
        """
        Parameters
        ----------
        shape : tuple of int
            Grid shape (ny, nx), excluding periodic points.
        coords : list of 4 floats
            [west (x), south (y), east (x), north (y)] domain boundaries.
        """
        assert len(coords) == 4, " coords must be a list of 4 floats: [west, south, east, north]."
        self.west, self.south, self.east, self.north = coords
        ny, nx = shape
        Ny, Nx = ny + 1, nx + 1

        self.x_coords, self.dx = np.linspace(
            self.west, self.east, Nx, retstep=True, dtype=np.float64
        )
        self.y_coords, self.dy = np.linspace(
            self.south, self.north, Ny, retstep=True, dtype=np.float64
        )

        ky = np.arange(ny)[:, None]
        kx = np.arange(nx // 2 + 1)[None, :]

        lam = (
            (2.0 * np.cos(2.0 * np.pi * kx / nx) - 2.0) / self.dx**2
            +
            (2.0 * np.cos(2.0 * np.pi * ky / ny) - 2.0) / self.dy**2
        )

        self.inv_lam = np.zeros_like(lam)
        self.inv_lam[lam != 0.0] = 1.0 / lam[lam != 0.0]
        self.inv_lam[0, 0] = 0.0  # Set the zero mode to zero to enforce zero mean solution
        self.shape = shape

    def solve(self, f):
        """
        Parameters
        ----------
        f : (ny, nx) ndarray
            Right-hand side, int_Omega f dx dy = 0.

        Returns
        -------
        u : (ny, nx) ndarray
            Solution with zero mean.
        """
        fhat = rfft2(f - f.mean())
        uhat = fhat * self.inv_lam
        return irfft2(uhat, s=self.shape)

def make_grid_points(A, B, C, D, nx, ny):
    """
    Construct periodic Cartesian grid points on rectangle ABCD.

    Returns
    -------
    grid_points : ndarray, shape (ny, nx, 2)
        grid_points[j, i] = physical coordinate [x, y]
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    D = np.asarray(D, dtype=float)

    xi = np.arange(nx) / nx
    eta = np.arange(ny) / ny

    XI, ETA = np.meshgrid(xi, eta, indexing="xy")

    grid_points = (
        A[None, None, :]
        + XI[..., None] * (B - A)[None, None, :]
        + ETA[..., None] * (D - A)[None, None, :]
    )

    return grid_points

def make_grid_points_mixed_bc(A, B, C, D, nx, ny):
    """
    Construct grid points for an x-periodic, y-Dirichlet channel.

    The y coordinate contains the ny interior points only, matching
    PoissonSolverChannel. The physical y=0 and y=H Dirichlet walls are
    excluded from the returned grid.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    D = np.asarray(D, dtype=float)

    xi = np.arange(nx) / nx
    eta = np.arange(1, ny + 1) / (ny + 1)

    XI, ETA = np.meshgrid(xi, eta, indexing="xy")

    return (
        A[None, None, :]
        + XI[..., None] * (B - A)[None, None, :]
        + ETA[..., None] * (D - A)[None, None, :]
    )

def poisson_gaussian_exact(X, Y, a, b, sigma, A=1.0, Nmodes=51):

    U = np.zeros_like(X, dtype=np.float64)

    # only odd modes contribute
    odd_modes = np.arange(1, Nmodes + 1, 2)

    for m in odd_modes:

        alpha_m = m * np.pi / a

        # exact Fourier coefficient integral in x
        z_m = (
            a / (2.0 * np.sqrt(2) * sigma)
            + 1j * alpha_m * sigma / np.sqrt(2)
        )

        Im = (
            np.sqrt(2 * np.pi)
            * sigma
            * np.exp(-0.5 * (alpha_m * sigma) ** 2)
            * np.real(erf(z_m))
            * np.sin(m * np.pi / 2)
        )

        sin_m = np.sin(m * np.pi * X / a)

        for n in odd_modes:

            beta_n = n * np.pi / b

            # exact Fourier coefficient integral in y
            z_n = (
                b / (2.0 * np.sqrt(2) * sigma)
                + 1j * beta_n * sigma / np.sqrt(2)
            )

            In = (
                np.sqrt(2 * np.pi)
                * sigma
                * np.exp(-0.5 * (beta_n * sigma) ** 2)
                * np.real(erf(z_n))
                * np.sin(n * np.pi / 2)
            )

            # Fourier coefficient
            Fmn = (4.0 * A / (a * b)) * Im * In

            # Poisson denominator
            denom = alpha_m**2 + beta_n**2

            # add mode contribution
            U += (
                Fmn / denom
            ) * sin_m * np.sin(n * np.pi * Y / b)

    return U

def select_grid_indices(grid_points, selector, order="lexsort"):
    """
    Select FFT-grid-aligned points using an arbitrary boolean expression.

    Parameters
    ----------
    grid_points : ndarray, shape (ny, nx, 2)
        Physical coordinates of FFT grid points.
    selector : callable
        Function selector(X, Y) -> boolean array of shape (ny, nx).
        X = grid_points[..., 0]
        Y = grid_points[..., 1]
    order : {"lexsort", "none"}
        Ordering of returned indices.

    Returns
    -------
    indices : ndarray, shape (N, 2)
        Each row is [j, i].
    """

    X = grid_points[..., 0]
    Y = grid_points[..., 1]
    mask = selector(X, Y)

    if mask.shape != X.shape:
        raise ValueError("selector must return a boolean array of shape (ny, nx).")

    indices = np.argwhere(mask)  # rows are [j, i]

    if order == "lexsort":
        # Sort primarily by j, then by i
        p = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[p]
    elif order == "none":
        pass
    else:
        raise ValueError("order must be 'lexsort' or 'none'.")

    return indices

def l_shape_boundary_indices(grid_points, x_cut, y_cut, tol=1e-11):
    X = grid_points[..., 0]
    Y = grid_points[..., 1]

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    removed_region = (X > x_cut) & (Y < y_cut)

    # outer_boundary = (
    #     (
    #         np.isclose(X, xmin, atol=tol)
    #         | np.isclose(Y, 1. - hy, atol=tol)
    #     )
    #     |
    #     (np.isclose(Y, ymin, atol=tol) & (X <= x_cut))
    #     |
    #     (np.isclose(X, 1. - hx, atol=tol) & (Y >= y_cut))
    # )
    c = 10
    outer_boundary = (
            np.isclose(X, c * hx, atol=tol)
            | np.isclose(Y, 1. - c * hy, atol=tol)
            | np.isclose(Y, c * hy, atol=tol)
            | np.isclose(X, 1. - c * hx, atol=tol)
        )

    inner_boundary = (
        (np.isclose(X, x_cut, atol=tol) & (Y <= y_cut))
        |
        (np.isclose(Y, y_cut, atol=tol) & (X >= x_cut))
    )

    # boundary_mask = (outer_boundary | inner_boundary) & ~removed_region
    boundary_mask = outer_boundary

    boundary_indices = np.argwhere(boundary_mask)

    return boundary_indices, boundary_mask

def poisson_periodic_obstacle_test_case(X, Y, Lx=1.0, Ly=1.0):

    kx1 = 2. * np.pi / Lx
    ky1 = 2. * np.pi / Ly
    kx2 = 4. * np.pi / Lx

    u_exact = (
        np.sin(kx1 * X) * np.cos(ky1 * Y)
        + 0.25 * np.cos(kx2 * X) * np.sin(ky1 * Y)
    )

    f = (
        (kx1**2 + ky1**2) * np.sin(kx1 * X) * np.cos(ky1 * Y)
        + 0.25 * (kx2**2 + ky1**2) * np.cos(kx2 * X) * np.sin(ky1 * Y)
    )

    return u_exact, -f

def rectangular_obstacle_mask(
    X,
    Y,
    Lx=1.0,
    Ly=1.0,
    width=0.25,
    height=0.25,
    x0=None,
    y0=None,
):

    x0 = Lx / 2 if x0 is None else x0
    y0 = Ly / 2 if y0 is None else y0
    x_vals = np.unique(X[0, :])
    y_vals = np.unique(Y[:, 0])
    local_dx = np.min(np.diff(x_vals)) if x_vals.size > 1 else Lx
    local_dy = np.min(np.diff(y_vals)) if y_vals.size > 1 else Ly
    tol = 0.5 * min(local_dx, local_dy)

    obstacle = (
        (X >= x0 - width/2) &
        (X <= x0 + width/2) &
        (Y >= y0 - height/2) &
        (Y <= y0 + height/2)
    )
    boundary = (Y > tol) & (
        (np.isclose(X, x0 - width/2, atol=tol) & (y0 - height/2 <= Y) & (Y <= y0 + height/2))
        | (np.isclose(Y, y0 + height/2, atol=tol) & (x0 - width/2 <= X) & (X <= x0 + width/2))
        | (np.isclose(Y, y0 - height/2, atol=tol) & (x0 - width/2 <= X) & (X <= x0 + width/2))
        | (np.isclose(X, x0 + width/2, atol=tol) & (y0 - height/2 <= Y) & (Y <= y0 + height/2))
    )
    # boundary = (
    #     np.isclose(X, int(width * nx) * hx, atol=tol)
    #     | np.isclose(Y, int(height * ny) * hy, atol=tol)
    #     | np.isclose(Y, int((1. - height) * ny) * hy, atol=tol)
    #     | np.isclose(X, int((1. - width) * nx) * hx, atol=tol)
    # )

    boundary_indices = np.argwhere(boundary)

    return obstacle, boundary_indices


def rectangular_corner_bump_mask(
    X,
    Y,
    west,
    south,
    east,
    north,
    width,
    height,
    corner="southeast",
):
    """Return a grid-aligned rectangular solid attached to a corner.

    The returned boundary indices contain only the faces exposed to fluid;
    the two faces coincident with the regular outer walls are omitted.
    """
    if corner != "southeast":
        raise NotImplementedError(
            "Only a southeast rectangular bump is currently supported."
        )
    if not 0.0 < width < east - west:
        raise ValueError("Bump width must lie inside the domain.")
    if not 0.0 < height < north - south:
        raise ValueError("Bump height must lie inside the domain.")

    X = np.asarray(X)
    Y = np.asarray(Y)
    x_left = east - width
    y_top = south + height
    x_values = np.unique(X[0, :])
    y_values = np.unique(Y[:, 0])
    dx = np.min(np.diff(x_values)) if x_values.size > 1 else width
    dy = np.min(np.diff(y_values)) if y_values.size > 1 else height
    tolerance = 0.25 * min(dx, dy)

    solid = (
        (X >= x_left - tolerance)
        & (Y <= y_top + tolerance)
    )
    exposed_west = (
        np.isclose(X, x_left, atol=tolerance)
        & (Y > south + tolerance)
        & (Y <= y_top + tolerance)
    )
    exposed_north = (
        np.isclose(Y, y_top, atol=tolerance)
        & (X >= x_left - tolerance)
    )
    return solid, np.argwhere(exposed_west | exposed_north)

def triangular_bump_mask(X, Y, Lx=1.0, Ly=1.0, width=0.25, height=0.25):
    """
    Triangular solid bump attached to the bottom boundary.
    """

    x0 = Lx / 2

    x_left  = x0 - width / 2
    x_right = x0 + width / 2

    inside_x = (X >= x_left) & (X <= x_right)

    # triangular top profile:
    # height at center, zero at x_left and x_right
    y_top = height * (1.0 - np.abs(X - x0) / (width / 2))

    bump = inside_x & (Y > 0.0) & (Y <= y_top)

    tol = 0.5 * min(dx, dy)

    left_edge = (
        (X >= x_left) &
        (X <= x0) &
        (Y > tol) &
        np.isclose(
            Y,
            height * (X - x_left) / (width/2),
            atol=tol
        )
    )

    right_edge = (
        (X >= x0) &
        (X <= x_right) &
        (Y > tol) &
        np.isclose(
            Y,
            height * (x_right - X) / (width/2),
            atol=tol
        )
    )

    boundary = left_edge | right_edge

    boundary_indices = np.argwhere(boundary)

    return bump, boundary_indices


if __name__ == "__main__":

    """ Convergence test """
    # Ns = [32, 64]
    # errors = [0.0021142332246262122, 0.0005266977162538242]

    # # for N in Ns:

    # #     err = u_final - u_exact
    # #     rms = np.sqrt(np.mean(err**2))
    # #     errors.append(rms)
    # #     print(N, rms)

    # for i in range(1, len(errors)):

    #     rate = np.log(errors[i-1] / errors[i]) / np.log(2)
    #     print(f"order ≈ {rate:.2f}")

    # raise ValueError
    """ ---------------- """

    nx, ny = 128, 96
    Lx, Ly = 2.0, 3.0
    hx, hy = Lx / nx, Ly / ny
    m, n = 3, 5

    sigma = 0.08

    A = np.array([0.0, 0.0])
    B = np.array([Lx, 0.0])
    C = np.array([Lx, Ly])
    D = np.array([0.0, Ly])

    grid_points = make_grid_points_mixed_bc(A, B, C, D, nx, ny)

    # Example irregular boundary points
    # theta = np.linspace(0., 2. * np.pi, 100, endpoint=False)
    # points = np.column_stack([
    #     1.0 + 0.4*np.cos(theta),
    #     0.5 + 0.2*np.sin(theta),
    # ])

    # boundary_indices = rectangle_boundary_indices(nx, ny)
    # def phi(X, Y):
    #     return ((X - 0.5 * Lx) / 0.5)**2 + ((Y - 0.5 * Ly) / 0.5)**2 - 1.0

    # boundary_indices = select_grid_indices(
    #     grid_points,
    #     selector=lambda X, Y: np.abs(phi(X, Y)) < 0.05
    # )

    # xcut, ycut = hx * 40, hy * 20
    # boundary_indices, boundary_mask = l_shape_boundary_indices(
    #     grid_points,
    #     x_cut=xcut,
    #     y_cut=ycut,
    # )

    # x = np.arange(nx) * hx
    # y = np.arange(ny) * hy
    # X, Y = np.meshgrid(x, y)

    # u_exact, f = poisson_periodic_obstacle_test_case(X, Y, Lx, Ly)

    solver = PoissonSolverChannel(shape=(ny, nx), coords=[0, 0, Lx, Ly])

    X, Y = np.meshgrid(solver.x_coords, solver.y_coords)

    u_exact = (
        np.sin(2 * np.pi * m * X / Lx)
        * np.sin(np.pi * n * Y / Ly)
    )

    dx, dy = solver.dx, solver.dy

    f = (
        (np.roll(u_exact, -1, axis=1) - 2*u_exact + np.roll(u_exact, 1, axis=1)) / dx**2
        +
        (
            np.vstack([u_exact[1:], np.zeros((1, nx))])
            - 2 * u_exact
            + np.vstack([np.zeros((1, nx)), u_exact[:-1]])
        ) / dy**2
    )

    # Define a source function
    def rho_func(x, y, a, b):
        # Example: Gaussian source around center
        # return np.exp(-4 * ((x - 1.) ** 2 + (y - 0.5) ** 2))
        return -np.exp(-((x - a/2)**2 + (y - b/2)**2) / (2 * sigma**2))

    f = rho_func(X, Y, Lx, Ly)

    # f = np.random.rand(ny, nx)
    u = solver.solve(f)

    # obstacle, boundary_indices = rectangular_obstacle_mask(
    #     X, Y,
    #     Lx=Lx,
    #     Ly=Ly,
    #     width=0.25,
    #     height=10 * dy,
    #     y0=0.1
    # )

    obstacle, boundary_indices = triangular_bump_mask(
        X, Y,
        Lx=Lx,
        Ly=Ly,
        width=0.75,
        height=0.75,
    )

    # boundary_indices = select_grid_indices(
    #     grid_points,
    #     selector=lambda X, Y: rectangular_obstacle_mask(
    #         X, Y,
    #         Lx=Lx,
    #         Ly=Ly,
    #         width=0.25,
    #         height=0.25
    #     )
    # )

    points = grid_points[
        boundary_indices[:, 0],
        boundary_indices[:, 1],
    ]

    # # plt.plot(grid_points[0, :], grid_points[1, :], 'ko', markersize=2)
    # plt.plot(points[:, 0], points[:, 1], "ro", markersize=5, markerfacecolor="none")
    # plt.xlim(0, Lx)
    # plt.ylim(0, Ly)
    # plt.show()

    # raise ValueError

    # Approximate panel weights
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)

    Cmat, info = build_capacitance_matrix_mixed_bc(
        A, B, C, D,
        points,
        nx,
        ny,
        weights=None,
    )

    # u_exact = poisson_gaussian_exact(
    #     X, Y,
    #     a=Lx,
    #     b=Ly,
    #     sigma=sigma,
    #     A=1.,
    #     Nmodes=51
    # )
    # f = (
    #     (np.roll(u_exact, -1, axis=1) - 2. * u_exact + np.roll(u_exact, 1, axis=1)) / hx**2
    #     +
    #     (np.roll(u_exact, -1, axis=0) - 2. * u_exact + np.roll(u_exact, 1, axis=0)) / hy**2
    # )

    u_periodic_boundary = u[
        boundary_indices[:, 0],
        boundary_indices[:, 1],
    ]
    g_boundary = u_exact[
        boundary_indices[:, 0],
        boundary_indices[:, 1],
    ]

    rhs = - u_periodic_boundary
    q = np.linalg.solve(Cmat, rhs)
    # A_aug = np.column_stack([Cmat, np.ones(Cmat.shape[0])])
    # sol, *_ = np.linalg.lstsq(A_aug, rhs, rcond=None)

    # q = sol[:-1]
    # c = sol[-1]

    grid_points2 = np.column_stack([X.ravel(), Y.ravel()])

    uc = apply_capacitance_correction_mixed_bc(
        grid_points2,
        points,
        q,
        # np.ones_like(ds),
        A, B, C, D,
        nx, ny,
        X.shape
    )

    u_final = u + uc.reshape(ny, nx) # + c

    # mask = phi(X, Y) <= 0.
    # mask = ~((grid_points[..., 0] > xcut) & (grid_points[..., 1] < ycut))
    # u_masked = np.where(mask, u_final, np.nan)

    # err = u - u_exact
    err = u_final[~obstacle] - u_exact[~obstacle]
    err -= err.mean()  # ignore arbitrary constant

    print("max error:", np.abs(err).max())
    print("rms error:", np.sqrt(np.mean(err**2)))

    u_plot = u_final.copy()
    u_plot[obstacle] = np.nan

    plt.imshow(u_plot, origin="lower", extent=(solver.west, solver.east, solver.south, solver.north))
    plt.colorbar()
    plt.title("Solution to Poisson's equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
