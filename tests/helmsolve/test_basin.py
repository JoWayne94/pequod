import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pequod.helmsolve.helmsolve import (
    PoissonSolverBasin,
    PoissonSolverBasinMasked,
    rectangular_corner_bump_mask,
)


def test_basin_poisson_inverts_the_discrete_laplacian():
    nx, ny = 13, 9
    solver = PoissonSolverBasin(
        shape=(ny, nx),
        coords=[-1.0, 0.25, 2.0, 1.75],
    )
    mode_x, mode_y = 3, 2
    x_mode = np.sin(
        np.pi * mode_x * np.arange(1, nx + 1) / (nx + 1)
    )
    y_mode = np.sin(
        np.pi * mode_y * np.arange(1, ny + 1) / (ny + 1)
    )
    exact = np.outer(y_mode, x_mode)
    eigenvalue = (
        (2.0 * np.cos(np.pi * mode_x / (nx + 1)) - 2.0)
        / solver.dx**2
        + (2.0 * np.cos(np.pi * mode_y / (ny + 1)) - 2.0)
        / solver.dy**2
    )

    np.testing.assert_allclose(
        solver.solve(eigenvalue * exact),
        exact,
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def test_masked_basin_poisson_enforces_southeast_bump():
    nx = ny = 24
    x = np.arange(nx) / nx
    y = np.arange(ny) / ny
    X, Y = np.meshgrid(x, y)
    solid, _ = rectangular_corner_bump_mask(
        X,
        Y,
        0.0,
        0.0,
        1.0,
        1.0,
        width=0.25,
        height=0.25,
    )
    solver = PoissonSolverBasinMasked(
        ~solid,
        coords=[0.0, 0.0, 1.0, 1.0],
    )
    rhs = np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)
    solution = solver.solve(rhs)

    laplacian = (
        (
            np.pad(solution, ((0, 0), (1, 1)))[:, 2:]
            - 2.0 * solution
            + np.pad(solution, ((0, 0), (1, 1)))[:, :-2]
        )
        / solver.dx**2
        + (
            np.pad(solution, ((1, 1), (0, 0)))[2:, :]
            - 2.0 * solution
            + np.pad(solution, ((1, 1), (0, 0)))[:-2, :]
        )
        / solver.dy**2
    )

    np.testing.assert_allclose(
        laplacian[solver.unknown_mask],
        rhs[solver.unknown_mask],
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert np.max(np.abs(solution[solid])) == 0.0
    assert np.max(np.abs(solution[0, :])) == 0.0
    assert np.max(np.abs(solution[:, 0])) == 0.0
