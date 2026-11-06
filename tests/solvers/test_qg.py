import sys
from functools import partial
import warnings
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pequod import grid, qg
qg = partial(qg, units="dimensionless")
from tutorials.qg.main import (
    make_basin_case,
    basin_manufactured_fields,
    channel_square_island_pv,
    channel_square_island_source,
    channel_square_island_streamfunction,
    channel_square_island_velocity,
    manufactured_square_hole_pv,
    manufactured_square_hole_source,
    manufactured_square_hole_streamfunction,
    manufactured_square_hole_velocity,
    periodic_beta_square_hole_source,
)


@pytest.mark.parametrize("case", ["basin_manufactured", "basin_northwest_manufactured"])
def test_basin_manufactured_inversion_and_evolution(case):
    inversion_errors, evolution_errors, pv_errors = [], [], []
    for n in (32, 64):
        solver, exact = make_basin_case(n, case)
        # Independently invert the continuum PV before advancing in time.
        for pre in solver.get_pre_processes():
            pre()
        mask = solver._active_mask("Basin")
        x, y = solver.X[:-1, :-1], solver.Y[:-1, :-1]
        psi, pv = (f(x, y, 0) for f in exact)
        inversion_errors.append(np.linalg.norm((solver.psi_c-psi)[mask]) /
                                np.linalg.norm(psi[mask]))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            while solver.t < solver.final_time - 1.e-15:
                for advance in solver.get_solve_sequence():
                    advance()
                solver.t += solver.dt
        evolution_errors.append(np.linalg.norm((solver.psi_c-psi)[mask]) /
                                np.linalg.norm(psi[mask]))
        pv_errors.append(np.linalg.norm((solver.potential_vorticity-pv)[mask]) /
                         np.linalg.norm(pv[mask]))
        assert np.max(abs(solver.psi_c[~mask])) == 0
        fx, fy = solver.physical_flux()
        assert np.max(abs(fx[..., [0, -1]])) == 0
        assert np.max(abs(fy[..., [0, -1], :])) == 0
        if solver.has_obstacle:
            assert np.max(abs(fx[:, solver.obstacle_faces_x])) == 0
            assert np.max(abs(fy[:, solver.obstacle_faces_y])) == 0
    assert inversion_errors[0] / inversion_errors[1] > 3.8
    assert evolution_errors[0] / evolution_errors[1] > 3.8
    assert pv_errors[1] < pv_errors[0]
    assert pv_errors[1] < 1.e-3


def test_northwest_bump_geometry_and_rendering_on_translated_basin():
    solver = qg(grid(32, 24, -2., 1., 2., 4.),
                obstacle="northwest_bump", obstacle_size=(1., .75))
    solver.domain.boundary_conditions = {"Basin": ((1., 0., 0.),)*4}
    x, y = solver.X[:-1, :-1], solver.Y[:-1, :-1]
    np.testing.assert_array_equal(solver.obstacle_mask, (x <= -1.) & (y >= 3.25))
    assert solver.obstacle_mask[-1, 0]
    assert not solver.obstacle_mask[0].any()
    assert not solver.obstacle_mask[:, -1].any()
    assert not solver.obstacle_faces_x[:, -1].any()  # no wrap to east
    assert not solver.obstacle_faces_y[0].any()     # no wrap to south
    plotted = solver.sol_plt(solver.psi_c)
    assert plotted.mask[-1, 0]
    assert plotted.mask[-1, 8]
    assert not plotted.mask[-1, 9]
    assert np.all((solver.points[:, 0] == -1.) | (solver.points[:, 1] == 3.25))


def test_northwest_bump_rejects_periodicity_and_unaligned_edges():
    solver = qg(grid(32, 32), obstacle="northwest_bump")
    with pytest.raises(NotImplementedError, match="Basin"):
        solver._validate_streamfunction_configuration("Periodic")
    with pytest.raises(ValueError, match="align"):
        qg(grid(32, 32), obstacle="northwest_bump", obstacle_size=(.2, .25))


def test_basin_northwest_bump_unforced_flow_evolves():
    solver, exact = make_basin_case(32)
    assert exact is None
    initial_pv = basin_manufactured_fields(
        solver.X[:-1, :-1], solver.Y[:-1, :-1], bump=True, amplitude=.01
    )[1]
    solver.final_time = .2
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver.solve(visual=-1)
    assert np.isfinite(solver.potential_vorticity).all()
    assert np.max(abs(solver.potential_vorticity)) > .1
    assert np.max(abs(solver.psi_c[solver.obstacle_mask])) == 0
    assert np.max(abs(solver.external_forcing)) == 0
    active = solver._active_mask("Basin")
    assert np.max(abs((solver.potential_vorticity-initial_pv)[active])) > 1.e-4


MANUFACTURED_AMPLITUDE = 1.0
MANUFACTURED_HALF_WIDTH = np.pi / 4.0
MANUFACTURED_DEFORMATION_RADIUS = 1.0


def _manufactured_pv(x, y):
    return manufactured_square_hole_pv(
        x,
        y,
        amplitude=MANUFACTURED_AMPLITUDE,
        half_width=MANUFACTURED_HALF_WIDTH,
        deformation_radius=MANUFACTURED_DEFORMATION_RADIUS,
    )


def _manufactured_velocity(x, y):
    return manufactured_square_hole_velocity(
        x,
        y,
        amplitude=MANUFACTURED_AMPLITUDE,
        half_width=MANUFACTURED_HALF_WIDTH,
    )


def _manufactured_solver(n):
    solver = qg(
        grid(n, n, -np.pi, -np.pi, np.pi, np.pi),
        obstacle="centered",
        obstacle_size=(
            2.0 * MANUFACTURED_HALF_WIDTH,
            2.0 * MANUFACTURED_HALF_WIDTH,
        ),
        deformation_radius=MANUFACTURED_DEFORMATION_RADIUS,
        beta=0.0,
        linear_drag=0.0,
    )
    solver.set_potential_vorticity_ivp(
        _manufactured_pv,
        _manufactured_velocity,
    )
    return solver


def _no_flow(x, y):
    return np.zeros_like(x), np.zeros_like(y)


def test_periodic_qg_helmholtz_inverts_a_discrete_mode():
    nx, ny = 18, 14
    length_x, length_y = 3.0, 2.0
    solver = qg(
        grid(nx, ny, 0.0, 0.0, length_x, length_y),
        obstacle=False,
        deformation_radius=0.8,
    )
    solver._ic = _no_flow
    solver._boundary_conditions()
    solver._laplacian_boundary_conditions()
    solver._psi_boundary_conditions()

    mode_x, mode_y = 3, 2
    x = np.arange(nx)
    y = np.arange(ny)
    exact = np.outer(
        np.cos(2.0 * np.pi * mode_y * y / ny),
        np.sin(2.0 * np.pi * mode_x * x / nx),
    )
    laplacian_eigenvalue = (
        (2.0 * np.cos(2.0 * np.pi * mode_x / nx) - 2.0)
        / solver.domain.dx**2
        + (2.0 * np.cos(2.0 * np.pi * mode_y / ny) - 2.0)
        / solver.domain.dy**2
    )
    solver.solutions[0, ...] = (
        laplacian_eigenvalue
        - solver.inverse_deformation_radius_squared
    ) * exact
    solver.compute_stream_functions()

    np.testing.assert_allclose(
        solver.psi_c,
        exact,
        atol=3.0e-13,
        rtol=3.0e-13,
    )


def test_periodic_qg_square_obstacle_remains_constrained():
    length = 2.0 * np.pi
    solver = qg(
        grid(32, 32, 0.0, 0.0, length, length),
        obstacle="centered",
        obstacle_size=(0.5 * np.pi, 0.5 * np.pi),
        deformation_radius=1.0,
        beta=0.0,
        linear_drag=0.01,
    )

    def periodic_vortex(x, y):
        xi = 2.0 * np.pi * x / length
        eta = 2.0 * np.pi * y / length
        amplitude = 0.35
        u = (
            -amplitude
            * (np.sin(xi) + 0.35 * np.cos(2.0 * xi + 0.3))
            * np.cos(eta)
        )
        v = (
            amplitude
            * (np.cos(xi) - 0.70 * np.sin(2.0 * xi + 0.3))
            * np.sin(eta)
        )
        return u, v

    solver.set_ivp(periodic_vortex)
    solver.cfl = 0.15
    solver.nu = 1.0e-3
    solver.final_time = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver.solve(visual=-1)

    flux_x, flux_y = solver.physical_flux()
    assert np.all(np.isfinite(solver.potential_vorticity))
    assert np.max(np.abs(solver.psi_c[solver.obstacle_mask])) == 0.0
    assert np.max(np.abs(flux_x[:, solver.obstacle_faces_x])) == 0.0
    assert np.max(np.abs(flux_y[:, solver.obstacle_faces_y])) == 0.0

    u = solver.adv_vel[0][0]
    v = solver.adv_vel[1][0]
    divergence = (
        (u - np.roll(u, 1, axis=-1)) / solver.domain.dx
        + (v - np.roll(v, 1, axis=-2)) / solver.domain.dy
    )
    assert np.max(np.abs(divergence)) < 2.0e-12

    laplacian = (
        (np.roll(solver.psi_c, -1, axis=1) - 2.0 * solver.psi_c
         + np.roll(solver.psi_c, 1, axis=1)) / solver.domain.dx**2
        + (np.roll(solver.psi_c, -1, axis=0) - 2.0 * solver.psi_c
           + np.roll(solver.psi_c, 1, axis=0)) / solver.domain.dy**2
    )
    active = ~solver.obstacle_mask
    np.testing.assert_allclose(
        solver.potential_vorticity[active],
        (
            laplacian
            - solver.inverse_deformation_radius_squared * solver.psi_c
        )[active],
        atol=3.0e-11,
        rtol=3.0e-11,
    )


@pytest.mark.parametrize(
    ("boundary_mode", "obstacle"),
    [
        ("Channel", "centered"),
        ("Basin", "southeast_bump"),
    ],
)
def test_qg_supports_closed_nonperiodic_boundaries(
    boundary_mode,
    obstacle,
):
    solver = qg(
        grid(32, 32, 0.0, 0.0, 1.0, 1.0),
        obstacle=obstacle,
        deformation_radius=1.0,
    )
    if boundary_mode == "Channel":
        solver.domain.boundary_conditions = {
            "Channel": (
                None,
                (1.0, 0.0, 0.0),
                None,
                (1.0, 0.0, 0.0),
            )
        }
    else:
        solver.domain.boundary_conditions = {
            "Basin": tuple((1.0, 0.0, 0.0) for _ in range(4))
        }
    solver.set_ivp(_no_flow)
    solver.cfl = 0.2
    solver.nu = 0.01
    solver.final_time = 0.05

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver.solve(visual=-1)

    assert np.max(np.abs(solver.potential_vorticity)) < 3.0e-15
    assert np.max(np.abs(solver.psi_c)) < 3.0e-15


def test_qg_beta_and_drag_sources_have_the_expected_sign():
    solver = qg(
        grid(8, 8, 0.0, 0.0, 1.0, 1.0),
        obstacle=False,
        deformation_radius=2.0,
        beta=1.5,
        linear_drag=0.2,
    )
    solver.external_forcing[...] = 0.7
    solver.cons_vel[1, 0, ...] = -0.4
    solver.solutions[0, ...] = 2.0
    solver.psi_c[...] = 0.8
    solver.update_dynamic_forcing()

    expected = 0.7 - 1.5 * (-0.4) - 0.2 * (2.0 + 0.8 / 4.0)
    np.testing.assert_allclose(solver.forcing, expected)


def test_manufactured_square_hole_pv_inversion_is_second_order():
    relative_errors = []
    for n in (32, 64):
        solver = _manufactured_solver(n)
        for preprocess in solver.get_pre_processes():
            preprocess()

        x = solver.X[:-1, :-1]
        y = solver.Y[:-1, :-1]
        exact = manufactured_square_hole_streamfunction(
            x,
            y,
            amplitude=MANUFACTURED_AMPLITUDE,
            half_width=MANUFACTURED_HALF_WIDTH,
        )
        fluid = ~solver.obstacle_mask
        error = np.linalg.norm((solver.psi_c - exact)[fluid])
        relative_errors.append(error / np.linalg.norm(exact[fluid]))

        assert np.max(np.abs(solver.psi_c[solver.obstacle_mask])) == 0.0
        flux_x, flux_y = solver.physical_flux()
        assert np.max(np.abs(flux_x[:, solver.obstacle_faces_x])) == 0.0
        assert np.max(np.abs(flux_y[:, solver.obstacle_faces_y])) == 0.0

    assert relative_errors[0] / relative_errors[1] > 3.9
    assert relative_errors[1] < 4.0e-4


def test_manufactured_square_hole_jacobian_and_periodicity():
    jacobian_errors = []
    for n in (32, 64):
        spacing = 2.0 * np.pi / n
        coordinates = -np.pi + spacing * np.arange(n)
        x, y = np.meshgrid(coordinates, coordinates)
        psi = manufactured_square_hole_streamfunction(x, y)
        pv = _manufactured_pv(x, y)
        source = manufactured_square_hole_source(x, y)

        derivative_x = lambda field: (  # noqa: E731
            np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)
        ) / (2.0 * spacing)
        derivative_y = lambda field: (  # noqa: E731
            np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)
        ) / (2.0 * spacing)
        numerical_jacobian = (
            derivative_x(psi) * derivative_y(pv)
            - derivative_y(psi) * derivative_x(pv)
        )
        jacobian_errors.append(
            np.linalg.norm(numerical_jacobian - source)
            / np.linalg.norm(source)
        )

        solver = _manufactured_solver(n)
        fluid = ~solver.obstacle_mask
        assert abs(np.sum(source[fluid]) * spacing**2) < 1.0e-14

    assert jacobian_errors[0] / jacobian_errors[1] > 3.9

    samples = np.linspace(-np.pi, np.pi, 33)
    np.testing.assert_allclose(
        manufactured_square_hole_streamfunction(-np.pi, samples),
        manufactured_square_hole_streamfunction(np.pi, samples),
        atol=1.0e-15,
    )
    u_west, v_west = _manufactured_velocity(-np.pi, samples)
    u_east, v_east = _manufactured_velocity(np.pi, samples)
    np.testing.assert_allclose(u_west, u_east, atol=1.0e-15)
    np.testing.assert_allclose(v_west, v_east, atol=1.0e-15)

    wall_samples = np.linspace(
        -MANUFACTURED_HALF_WIDTH,
        MANUFACTURED_HALF_WIDTH,
        17,
    )
    u_west, _ = _manufactured_velocity(
        -MANUFACTURED_HALF_WIDTH,
        wall_samples,
    )
    u_east, _ = _manufactured_velocity(
        MANUFACTURED_HALF_WIDTH,
        wall_samples,
    )
    _, v_south = _manufactured_velocity(
        wall_samples,
        -MANUFACTURED_HALF_WIDTH,
    )
    _, v_north = _manufactured_velocity(
        wall_samples,
        MANUFACTURED_HALF_WIDTH,
    )
    np.testing.assert_allclose(u_west, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(u_east, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(v_south, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(v_north, 0.0, atol=1.0e-15)


def test_manufactured_square_hole_remains_nearly_steady():
    errors = []
    for n in (32, 64):
        solver = _manufactured_solver(n)
        solver.set_forcing_terms(manufactured_square_hole_source)
        solver.cfl = 0.15
        solver.nu = 0.0
        solver.final_time = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            solver.solve(visual=-1)

        x = solver.X[:-1, :-1]
        y = solver.Y[:-1, :-1]
        fluid = ~solver.obstacle_mask
        exact_q = _manufactured_pv(x, y)
        exact_psi = manufactured_square_hole_streamfunction(x, y)
        q_error = np.linalg.norm(
            (solver.potential_vorticity - exact_q)[fluid]
        ) / np.linalg.norm(exact_q[fluid])
        psi_error = np.linalg.norm(
            (solver.psi_c - exact_psi)[fluid]
        ) / np.linalg.norm(exact_psi[fluid])
        errors.append((q_error, psi_error))

    assert errors[1][0] < errors[0][0]
    assert errors[0][1] / errors[1][1] > 3.9
    assert errors[1][0] < 5.0e-3
    assert errors[1][1] < 4.0e-4


def test_periodic_beta_square_hole_manufactured_solution_converges():
    psi_errors = []
    q_errors = []
    for n in (32, 64):
        solver = _manufactured_solver(n)
        solver.beta = 1.0
        solver.set_forcing_terms(periodic_beta_square_hole_source)
        solver.cfl = 0.15
        solver.nu = 0.0
        solver.final_time = 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            solver.solve(visual=-1)

        x = solver.X[:-1, :-1]
        y = solver.Y[:-1, :-1]
        fluid = ~solver.obstacle_mask
        exact_psi = manufactured_square_hole_streamfunction(x, y)
        exact_q = _manufactured_pv(x, y)
        psi_errors.append(
            np.linalg.norm((solver.psi_c - exact_psi)[fluid])
            / np.linalg.norm(exact_psi[fluid])
        )
        q_errors.append(
            np.linalg.norm(
                (solver.potential_vorticity - exact_q)[fluid]
            ) / np.linalg.norm(exact_q[fluid])
        )
        flux_x, flux_y = solver.physical_flux()
        np.testing.assert_allclose(
            flux_x[:, solver.obstacle_faces_x],
            0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            flux_y[:, solver.obstacle_faces_y],
            0.0,
            atol=0.0,
        )
        forcing_integral = (
            np.sum(solver.external_forcing[fluid])
            * solver.domain.dx
            * solver.domain.dy
        )
        assert abs(forcing_integral) < 1.0e-14

    assert psi_errors[0] / psi_errors[1] > 3.9
    assert q_errors[1] < q_errors[0]


def test_channel_square_island_manufactured_solution_converges():
    parameters = {
        "amplitude": 1.0,
        "length_x": 2.0 * np.pi,
        "length_y": np.pi,
        "island_half_width": np.pi / 4.0,
        "island_half_height": np.pi / 4.0,
        "island_center_x": np.pi,
        "beta": 1.0,
        "deformation_radius": 1.0,
    }
    psi_errors = []
    q_errors = []
    for n in (32, 64):
        solver = qg(
            grid(n, n, 0.0, -0.5 * np.pi, 2.0 * np.pi, 0.5 * np.pi),
            obstacle="centered",
            obstacle_size=(0.5 * np.pi, 0.5 * np.pi),
            deformation_radius=parameters["deformation_radius"],
            beta=parameters["beta"],
        )
        stationary_slip = (1.0, 0.0, 0.0)
        solver.domain.boundary_conditions = {
            "Channel": (None, stationary_slip, None, stationary_slip)
        }
        solver.set_potential_vorticity_ivp(
            lambda x, y: channel_square_island_pv(
                x,
                y,
                **parameters,
            ),
            lambda x, y: channel_square_island_velocity(
                x,
                y,
                **parameters,
            ),
        )
        solver.set_forcing_terms(
            lambda x, y: channel_square_island_source(
                x,
                y,
                **parameters,
            )
        )
        solver.cfl = 0.15
        solver.nu = 0.0
        solver.final_time = 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            solver.solve(visual=-1)

        x = solver.X[:-1, :-1]
        y = solver.Y[:-1, :-1]
        fluid = solver._active_mask("Channel")
        exact_psi = channel_square_island_streamfunction(
            x, y, **parameters
        )
        exact_q = channel_square_island_pv(x, y, **parameters)
        psi_errors.append(
            np.linalg.norm((solver.psi_c - exact_psi)[fluid])
            / np.linalg.norm(exact_psi[fluid])
        )
        q_errors.append(
            np.linalg.norm(
                (solver.potential_vorticity - exact_q)[fluid]
            ) / np.linalg.norm(exact_q[fluid])
        )
        flux_x, flux_y = solver.physical_flux()
        np.testing.assert_allclose(flux_y[..., 0, :], 0.0, atol=0.0)
        np.testing.assert_allclose(flux_y[..., -1, :], 0.0, atol=0.0)
        np.testing.assert_allclose(
            flux_x[:, solver.obstacle_faces_x],
            0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            flux_y[:, solver.obstacle_faces_y],
            0.0,
            atol=0.0,
        )
        assert np.max(np.abs(solver.psi_c[solver.obstacle_mask])) == 0.0
        forcing_integral = (
            np.sum(solver.external_forcing[fluid])
            * solver.domain.dx
            * solver.domain.dy
        )
        assert abs(forcing_integral) < 1.0e-14

    assert psi_errors[0] / psi_errors[1] > 3.9
    assert q_errors[0] / q_errors[1] > 1.3

    wall_x = np.linspace(0.0, 2.0 * np.pi, 65, endpoint=False)
    np.testing.assert_allclose(
        channel_square_island_streamfunction(
            wall_x,
            -0.5 * np.pi,
            **parameters,
        ),
        0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        channel_square_island_streamfunction(
            wall_x,
            0.5 * np.pi,
            **parameters,
        ),
        0.0,
        atol=1.0e-15,
    )
    island_coordinate = np.linspace(-0.25 * np.pi, 0.25 * np.pi, 33)
    np.testing.assert_allclose(
        channel_square_island_streamfunction(
            np.pi - 0.25 * np.pi,
            island_coordinate,
            **parameters,
        ),
        0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        channel_square_island_streamfunction(
            np.pi + 0.25 * np.pi,
            island_coordinate,
            **parameters,
        ),
        0.0,
        atol=1.0e-15,
    )
    island_x = np.linspace(
        np.pi - 0.25 * np.pi,
        np.pi + 0.25 * np.pi,
        33,
    )
    np.testing.assert_allclose(
        channel_square_island_streamfunction(
            island_x,
            -0.25 * np.pi,
            **parameters,
        ),
        0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        channel_square_island_streamfunction(
            island_x,
            0.25 * np.pi,
            **parameters,
        ),
        0.0,
        atol=1.0e-15,
    )


def test_channel_square_island_exact_fields_satisfy_qg_equations():
    parameters = {
        "amplitude": 1.0,
        "length_x": 2.0 * np.pi,
        "length_y": np.pi,
        "island_half_width": np.pi / 4.0,
        "island_half_height": np.pi / 4.0,
        "island_center_x": np.pi,
        "beta": 1.0,
        "deformation_radius": 1.0,
    }
    q_errors = []
    source_errors = []
    for n in (64, 128):
        dx = parameters["length_x"] / n
        dy = parameters["length_y"] / n
        x_values = dx * np.arange(n)
        y_values = -0.5 * np.pi + dy * np.arange(n + 1)
        x, y = np.meshgrid(x_values, y_values)
        psi = channel_square_island_streamfunction(x, y, **parameters)
        exact_q = channel_square_island_pv(x, y, **parameters)
        exact_source = channel_square_island_source(x, y, **parameters)

        derivative_x = lambda field: (  # noqa: E731
            np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)
        ) / (2.0 * dx)
        derivative_y = lambda field: (  # noqa: E731
            field[2:, :] - field[:-2, :]
        ) / (2.0 * dy)
        laplacian_x = (
            np.roll(psi, -1, axis=1)
            - 2.0 * psi
            + np.roll(psi, 1, axis=1)
        ) / dx**2
        laplacian_y = (
            psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]
        ) / dy**2
        numerical_q = (
            laplacian_x[1:-1, :]
            + laplacian_y
            - psi[1:-1, :] / parameters["deformation_radius"] ** 2
        )
        q_errors.append(
            np.linalg.norm(numerical_q - exact_q[1:-1, :])
            / np.linalg.norm(exact_q[1:-1, :])
        )

        psi_x = derivative_x(psi)[1:-1, :]
        psi_y = derivative_y(psi)
        q_x = derivative_x(exact_q)[1:-1, :]
        q_y = derivative_y(exact_q)
        numerical_source = (
            psi_x * q_y
            - psi_y * q_x
            + parameters["beta"] * psi_x
        )
        source_errors.append(
            np.linalg.norm(numerical_source - exact_source[1:-1, :])
            / np.linalg.norm(exact_source[1:-1, :])
        )

    assert q_errors[0] / q_errors[1] > 3.9
    assert source_errors[0] / source_errors[1] > 3.9
