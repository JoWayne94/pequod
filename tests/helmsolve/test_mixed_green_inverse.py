import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pequod.CMM.ccm_test import mixed_green_matrix_cross
from pequod.helmsolve.helmsolve import PoissonSolverChannel


def test_mixed_green_matches_channel_poisson_inverse():
    nx, ny = 12, 7
    west, south, east, north = -0.5, 0.25, 2.5, 1.75
    length_x = east - west
    length_y = north - south

    solver = PoissonSolverChannel(
        shape=(ny, nx),
        coords=[west, south, east, north],
    )
    source_row, source_col = 3, 5
    rhs = np.zeros((ny, nx))
    rhs[source_row, source_col] = 1.0

    x_local = solver.x_coords - west
    y_local = solver.y_coords - south
    x_grid, y_grid = np.meshgrid(x_local, y_local)
    evaluation_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    source_point = np.array(
        [[x_local[source_col], y_local[source_row]]]
    )

    green_column = mixed_green_matrix_cross(
        evaluation_points,
        source_point,
        length_x,
        length_y,
        nx,
        ny,
    ).reshape(ny, nx)

    np.testing.assert_allclose(
        green_column,
        -solver.solve(rhs),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
