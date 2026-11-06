import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pequod import grid, incns


def _no_flow(x, y):
    return np.zeros_like(x), np.zeros_like(y)


def _channel_solver(n, *, obstacle):
    solver = incns(
        grid(n, n, 0.0, 0.0, 1.0, 1.0),
        obstacle=obstacle,
    )
    solver.domain.boundary_conditions = {
        "Channel": (
            None,
            (1.0, 0.0, 0.0),
            None,
            (1.0, 0.0, 0.0),
        )
    }
    solver.set_ivp(_no_flow)
    return solver


def _laminar_vorticity_error(n):
    force_amplitude = 0.1
    viscosity = 0.05
    final_time = 0.1
    wave_number = 2.0 * np.pi

    solver = _channel_solver(n, obstacle=False)
    solver.set_forcing_terms(
        lambda x, y: (
            -force_amplitude * wave_number * np.cos(wave_number * y)
        )
    )
    solver.cfl = 0.1
    solver.nu = viscosity
    solver.final_time = final_time
    solver.solve(visual=-1)

    amplitude = (
        force_amplitude
        / (viscosity * wave_number**2)
        * (1.0 - np.exp(-viscosity * wave_number**2 * final_time))
    )
    omega_exact = (
        -wave_number
        * amplitude
        * np.cos(wave_number * solver.Y[:-1, :-1])
    )
    return np.sqrt(np.mean((solver.solutions[0] - omega_exact) ** 2))


def test_laminar_channel_converges_at_second_order():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        coarse_error = _laminar_vorticity_error(16)
        fine_error = _laminar_vorticity_error(32)

    assert fine_error < 7.0e-5
    assert coarse_error / fine_error > 3.5


def test_obstacle_projection_and_fluxes_remain_constrained():
    solver = _channel_solver(32, obstacle=True)
    solver.set_forcing_terms(
        lambda x, y: -0.1 * 2.0 * np.pi * np.cos(2.0 * np.pi * y)
    )
    solver.cfl = 0.1
    solver.nu = 0.05
    solver.final_time = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver.solve(visual=-1)

    rows = solver.boundary_indices[:, 0]
    cols = solver.boundary_indices[:, 1]
    flux_x, flux_y = solver.physical_flux()

    assert np.all(np.isfinite(solver.solutions))
    assert np.max(np.abs(solver.psi_c[rows, cols])) < 1.0e-12
    assert np.max(np.abs(flux_x[:, solver.obstacle_faces_x])) == 0.0
    assert np.max(np.abs(flux_y[:, solver.obstacle_faces_y])) == 0.0


def test_channel_throughflow_preserves_the_harmonic_velocity():
    speed = 1.25
    solver = incns(
        grid(32, 32, 0.0, 0.0, 1.0, 1.0),
        obstacle=False,
    )
    solver.channel_throughflow = speed
    solver.domain.boundary_conditions = {
        "Channel": (
            None,
            (1.0, 0.0, speed),
            None,
            (1.0, 0.0, speed),
        )
    }
    solver.set_ivp(
        lambda x, y: (speed * np.ones_like(x), np.zeros_like(y))
    )
    solver.cfl = 0.2
    solver.nu = 0.01
    solver.final_time = 0.05

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver.solve(visual=-1)

    expected_psi = -speed * solver.Y[:-1, :-1]
    np.testing.assert_allclose(solver.psi_c, expected_psi, atol=2.0e-13)
    np.testing.assert_allclose(solver.adv_vel[0], speed, atol=2.0e-12)
    np.testing.assert_allclose(solver.adv_vel[1], 0.0, atol=2.0e-12)


def test_channel_throughflow_uses_a_constant_obstacle_streamfunction():
    speed = 1.0
    solver = _channel_solver(32, obstacle=True)
    solver.channel_throughflow = speed

    def poiseuille(x, y):
        eta = y
        return 6.0 * speed * eta * (1.0 - eta), np.zeros_like(y)

    solver.set_ivp(poiseuille)
    for preprocess in solver.get_pre_processes():
        preprocess()

    rows = solver.boundary_indices[:, 0]
    cols = solver.boundary_indices[:, 1]
    np.testing.assert_allclose(
        solver.psi_c[rows, cols],
        solver._obstacle_streamfunction,
        atol=2.0e-13,
    )


def test_basin_with_southeast_bump_remains_closed_and_finite():
    solver = incns(
        grid(32, 32, 0.0, 0.0, 1.0, 1.0),
        obstacle="southeast_bump",
        obstacle_size=(0.25, 0.25),
    )
    solver.domain.boundary_conditions = {
        "Basin": tuple((1.0, 0.0, 0.0) for _ in range(4))
    }

    def basin_vortex(x, y):
        u = (
            -2.0
            * np.pi
            * np.sin(np.pi * x) ** 2
            * np.sin(np.pi * y)
            * np.cos(np.pi * y)
        )
        v = (
            2.0
            * np.pi
            * np.sin(np.pi * x)
            * np.cos(np.pi * x)
            * np.sin(np.pi * y) ** 2
        )
        return u, v

    solver.set_ivp(basin_vortex)
    solver.cfl = 0.1
    solver.nu = 0.01
    solver.final_time = 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver.solve(visual=-1)

    flux_x, flux_y = solver.physical_flux()
    assert np.all(np.isfinite(solver.solutions))
    assert np.max(np.abs(solver.psi_c[0, :])) == 0.0
    assert np.max(np.abs(solver.psi_c[:, 0])) == 0.0
    assert np.max(np.abs(solver.psi_c[solver.obstacle_mask])) == 0.0
    assert np.max(np.abs(flux_x[:, solver.obstacle_faces_x])) == 0.0
    assert np.max(np.abs(flux_y[:, solver.obstacle_faces_y])) == 0.0

    # The solid reaches both regular boundaries in the stored numerical
    # geometry and in the north/east-padded plotting geometry.
    assert solver.obstacle_mask[0, -1]
    south_solid = np.flatnonzero(solver.obstacle_mask[0, :])
    east_solid = np.flatnonzero(solver.obstacle_mask[:, -1])
    np.testing.assert_array_equal(
        south_solid,
        np.arange(south_solid[0], solver.domain.nx),
    )
    np.testing.assert_array_equal(
        east_solid,
        np.arange(east_solid[-1] + 1),
    )
    plotted_mask = np.ma.getmaskarray(solver.sol_plt(solver.psi_c))
    assert plotted_mask[0, -1]
    assert np.all(plotted_mask[0, south_solid[0]:])
    assert np.all(plotted_mask[:east_solid[-1] + 1, -1])

    dx2 = solver.domain.dx**2
    dy2 = solver.domain.dy**2
    south_fluid = ~solver.obstacle_mask[0, 1:-1]
    north_fluid = ~solver.obstacle_mask[-1, 1:-1]
    west_fluid = ~solver.obstacle_mask[1:-1, 0]
    east_fluid = ~solver.obstacle_mask[1:-1, -1]
    np.testing.assert_allclose(
        solver.solutions[0, 0, 1:-1][south_fluid],
        2.0 * solver.psi_c[1, 1:-1][south_fluid] / dy2,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        solver._adv_y[0, 0, -1, 1:-1][north_fluid],
        2.0 * solver.psi_c[-1, 1:-1][north_fluid] / dy2,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        solver.solutions[0, 1:-1, 0][west_fluid],
        2.0 * solver.psi_c[1:-1, 1][west_fluid] / dx2,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        solver._adv_x[0, 0, 1:-1, -1][east_fluid],
        2.0 * solver.psi_c[1:-1, -1][east_fluid] / dx2,
        atol=2.0e-12,
    )

    u = solver.adv_vel[0][0]
    v = solver.adv_vel[1][0]
    divergence = (
        (u - np.roll(u, 1, axis=-1)) / solver.domain.dx
        + (v - np.roll(v, 1, axis=-2)) / solver.domain.dy
    )
    fluid_interior = (~solver.obstacle_mask)[1:, 1:]
    assert np.max(np.abs(divergence[1:, 1:][fluid_interior])) < 1.0e-11
