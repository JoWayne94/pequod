import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pequod.helmsolve.helmsolve import (
    PoissonSolverChannel,
    build_capacitance_matrix_mixed_bc,
    make_grid_points_mixed_bc,
    rectangular_obstacle_mask,
)


def test_mixed_bc_capacitance_matrix_uses_interior_y_grid():
    nx, ny = 32, 24
    Lx, Ly = 2.0, 3.0
    A = np.array([0.0, 0.0])
    B = np.array([Lx, 0.0])
    C = np.array([Lx, Ly])
    D = np.array([0.0, Ly])

    solver = PoissonSolverChannel(shape=(ny, nx), coords=[0.0, 0.0, Lx, Ly])
    X, Y = np.meshgrid(solver.x_coords, solver.y_coords)
    grid_points = make_grid_points_mixed_bc(A, B, C, D, nx, ny)

    _, boundary_indices = rectangular_obstacle_mask(
        X,
        Y,
        Lx=Lx,
        Ly=Ly,
        width=0.25,
        height=4.0 * solver.dy,
        y0=0.3,
    )
    points = grid_points[boundary_indices[:, 0], boundary_indices[:, 1]]

    assert points.shape[0] > 0
    assert np.all(points[:, 1] > 0.0)
    assert np.all(points[:, 1] < Ly)

    Cmat, _ = build_capacitance_matrix_mixed_bc(
        A,
        B,
        C,
        D,
        points,
        nx,
        ny,
        weights=None,
    )

    assert np.linalg.matrix_rank(Cmat) == Cmat.shape[0]
