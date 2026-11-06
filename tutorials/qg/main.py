"""Single-layer QG manufactured and square-obstacle benchmarks."""

import argparse
from functools import lru_cache, partial
from numpy.polynomial import Polynomial

from pequod import grid, qg
import numpy as np

# Manufactured solutions and the legacy tutorial parameters use model units.
qg = partial(qg, units="dimensionless")


def manufactured_square_hole_streamfunction(
    x,
    y,
    *,
    amplitude=1.0,
    half_width=np.pi / 4.0,
):
    r"""Exact streamfunction for the periodic square-hole benchmark.

    ``psi=A(cos(x)-cos(a))(cos(y)-cos(a))`` is zero on every face of
    ``[-a,a]^2`` and is exactly 2*pi-periodic in both directions.
    """
    c = np.cos(half_width)
    return amplitude * (np.cos(x) - c) * (np.cos(y) - c)


def manufactured_square_hole_pv(
    x,
    y,
    *,
    amplitude=1.0,
    half_width=np.pi / 4.0,
    deformation_radius=1.0,
):
    """Exact ``q=Laplacian(psi)-psi/L_D**2`` for the benchmark."""
    c = np.cos(half_width)
    cosine_x = np.cos(x)
    cosine_y = np.cos(y)
    inverse_radius_squared = 1.0 / deformation_radius**2
    return -amplitude * (
        cosine_x * (cosine_y - c)
        + cosine_y * (cosine_x - c)
        + inverse_radius_squared
        * (cosine_x - c)
        * (cosine_y - c)
    )


def manufactured_square_hole_velocity(
    x,
    y,
    *,
    amplitude=1.0,
    half_width=np.pi / 4.0,
):
    """Exact velocity ``(u,v)=(-psi_y,psi_x)`` for the benchmark."""
    c = np.cos(half_width)
    u = amplitude * (np.cos(x) - c) * np.sin(y)
    v = -amplitude * np.sin(x) * (np.cos(y) - c)
    return u, v


def manufactured_square_hole_source(
    x,
    y,
    *,
    amplitude=1.0,
    half_width=np.pi / 4.0,
):
    r"""Exact source ``S=J(psi,q)``; independent of deformation radius.

    For ``J=psi_x*q_y-psi_y*q_x``, the final factor is
    ``cos(x)-cos(y)``.  Reversing that factor gives ``-J``.
    """
    return (
        amplitude**2
        * np.cos(half_width)
        * np.sin(x)
        * np.sin(y)
        * (np.cos(x) - np.cos(y))
    )


def periodic_beta_square_hole_source(
    x,
    y,
    *,
    amplitude=1.0,
    half_width=np.pi / 4.0,
    beta=1.0,
):
    r"""Return ``J(psi_e,q_e)+beta*psi_e,x`` on the periodic square hole."""
    c = np.cos(half_width)
    jacobian = manufactured_square_hole_source(
        x,
        y,
        amplitude=amplitude,
        half_width=half_width,
    )
    psi_x = -amplitude * np.sin(x) * (np.cos(y) - c)
    return jacobian + beta * psi_x


def _channel_square_island_fields(
    x,
    y,
    *,
    amplitude=1.0,
    length_x=2.0 * np.pi,
    length_y=np.pi,
    island_half_width=np.pi / 4.0,
    island_half_height=np.pi / 4.0,
    island_center_x=np.pi,
    deformation_radius=1.0,
    beta=1.0,
):
    r"""Evaluate the exact channel-island fields and their derivatives.

    The manufactured streamfunction is

    ``psi=A sin^2(pi*(y+Ly/2)/Ly)``
    ``       [cos(kx*(x-x0))-cos(kx*a_x)]``
    ``       [cos(ky*y)-cos(ky*a_y)]``,

    where ``kx=2*pi/Lx`` and ``ky=2*pi/Ly``.  It vanishes at both channel
    walls and on every face of the centered rectangular island.
    """
    zonal_wave = 2.0 * np.pi / length_x
    meridional_wave = 2.0 * np.pi / length_y
    wall_wave = np.pi / length_y
    inverse_radius_squared = deformation_radius**-2

    phase_x = zonal_wave * (x - island_center_x)
    f = np.cos(phase_x) - np.cos(zonal_wave * island_half_width)
    f_x = -zonal_wave * np.sin(phase_x)
    f_xx = -zonal_wave**2 * np.cos(phase_x)
    f_xxx = zonal_wave**3 * np.sin(phase_x)

    phase_wall = wall_wave * (y + 0.5 * length_y)
    wall = np.sin(phase_wall) ** 2
    wall_y = wall_wave * np.sin(2.0 * phase_wall)
    wall_yy = 2.0 * wall_wave**2 * np.cos(2.0 * phase_wall)
    wall_yyy = -4.0 * wall_wave**3 * np.sin(2.0 * phase_wall)

    phase_y = meridional_wave * y
    g = np.cos(phase_y) - np.cos(
        meridional_wave * island_half_height
    )
    g_y = -meridional_wave * np.sin(phase_y)
    g_yy = -meridional_wave**2 * np.cos(phase_y)
    g_yyy = meridional_wave**3 * np.sin(phase_y)

    h = wall * g
    h_y = wall_y * g + wall * g_y
    h_yy = wall_yy * g + 2.0 * wall_y * g_y + wall * g_yy
    h_yyy = (
        wall_yyy * g
        + 3.0 * wall_yy * g_y
        + 3.0 * wall_y * g_yy
        + wall * g_yyy
    )
    helmholtz_y = h_yy - inverse_radius_squared * h
    helmholtz_y_derivative = (
        h_yyy - inverse_radius_squared * h_y
    )

    psi = amplitude * f * h
    psi_x = amplitude * f_x * h
    psi_y = amplitude * f * h_y
    q = amplitude * (f_xx * h + f * helmholtz_y)
    q_x = amplitude * (f_xxx * h + f_x * helmholtz_y)
    q_y = amplitude * (
        f_xx * h_y + f * helmholtz_y_derivative
    )
    source = psi_x * q_y - psi_y * q_x + beta * psi_x
    return psi, q, -psi_y, psi_x, source


def channel_square_island_streamfunction(x, y, **parameters):
    """Exact streamfunction for the periodic-x channel island benchmark."""
    return _channel_square_island_fields(x, y, **parameters)[0]


def channel_square_island_pv(x, y, **parameters):
    """Exact PV for the periodic-x channel island benchmark."""
    return _channel_square_island_fields(x, y, **parameters)[1]


def channel_square_island_velocity(x, y, **parameters):
    """Exact velocity for the periodic-x channel island benchmark."""
    fields = _channel_square_island_fields(x, y, **parameters)
    return fields[2], fields[3]


def channel_square_island_source(x, y, **parameters):
    r"""Exact source ``J(psi_e,q_e)+beta*psi_e,x`` for the channel."""
    return _channel_square_island_fields(x, y, **parameters)[4]


def relative_l2_error(numerical, exact, mask):
    """Return a discrete relative L2 error over masked fluid points."""
    difference = numerical[mask] - exact[mask]
    return np.linalg.norm(difference) / np.linalg.norm(exact[mask])


@lru_cache(maxsize=2)
def _basin_polynomials(bump):
    """Normalize the separable manufactured polynomials once per geometry."""
    factors = []
    for root in ((0.25 if bump else None), (0.75 if bump else None)):
        p = Polynomial([0., 1.])**3 * Polynomial([1., -1.])**3
        if root is not None:
            p *= Polynomial([-root, 1.])**3
        extrema = p.deriv().roots()
        extrema = extrema.real[(abs(extrema.imag) < 1.e-10)
                               & (extrema.real >= 0) & (extrema.real <= 1)]
        p /= np.max(np.abs(p(extrema)))
        factors.append(p)
    return tuple(factors)


def basin_manufactured_fields(x, y, *, bump=False, beta=1.0, amplitude=1.0):
    """Steady QG fields on [0,1]^2, optionally with a NW quarter-size bump.

    psi=A*f(x)*g(y); f and g have cubic zeros at regular walls and,
    for the bump, at x=1/4 and y=3/4. Thus psi and its first two
    derivatives vanish on each solid face. L_D=1 and nu=drag=0.
    Each polynomial is normalized by its maximum absolute value on [0,1].
    Returns psi, q, u, v, S=J(psi,q)+beta*v.
    """
    f, g = _basin_polynomials(bump)
    psi = amplitude * f(x) * g(y)
    u = -amplitude * f(x) * g.deriv()(y)
    v = amplitude * f.deriv()(x) * g(y)
    q = amplitude * (f.deriv(2)(x)*g(y) + f(x)*g.deriv(2)(y)) - psi
    qx = amplitude * (f.deriv(3)(x)*g(y) + f.deriv()(x)*g.deriv(2)(y)) - v
    qy = amplitude * (f.deriv(2)(x)*g.deriv()(y) + f(x)*g.deriv(3)(y)) + u
    return psi, q, u, v, u*qx + v*qy + beta*v


def make_basin_case(n=128, case="basin_northwest_bump"):
    """Build a closed basin benchmark or an unforced NW-bump vortex decay.

    The solid is [0,1/4] x [3/4,1], attached to west and north.
    The dynamic case starts with the same compatible PV/velocity as the
    manufactured bump case, removes its sustaining source and adds drag.
    """
    if case not in {"basin_manufactured", "basin_northwest_manufactured",
                    "basin_northwest_bump"}:
        raise ValueError(f"Unknown basin case: {case}")
    bump = case != "basin_manufactured"
    dynamic = case == "basin_northwest_bump"
    solver = qg(grid(n, n, 0., 0., 1., 1.),
                obstacle="northwest_bump" if bump else False,
                obstacle_size=(0.25, 0.25), beta=1.,
                deformation_radius=1., linear_drag=0.01 if dynamic else 0.)
    solver.domain.boundary_conditions = {"Basin": ((1., 0., 0.),)*4}
    fields = lambda x, y: basin_manufactured_fields(x, y, bump=bump, amplitude=0.01)
    solver.set_potential_vorticity_ivp(
        lambda x, y: fields(x, y)[1],
        lambda x, y: fields(x, y)[2:4],
    )
    if not dynamic:
        solver.set_forcing_terms(lambda x, y: fields(x, y)[4])
    solver.nu = 0.
    solver.cfl = 0.15
    solver.final_time = 2. if dynamic else 0.1
    exact = None if dynamic else (
        lambda x, y, t: fields(x, y)[0],
        lambda x, y, t: fields(x, y)[1],
    )
    return solver, exact


def periodic_square_cylinder_vortex(
    x,
    y,
    *,
    x_left=0.0,
    x_right=2.0 * np.pi,
    y_bottom=0.0,
    y_top=2.0 * np.pi,
    amplitude=0.35,
):
    r"""Velocity IVP generated by an asymmetric periodic streamfunction.

    With ``xi=2*pi*(x-x_left)/Lx`` and
    ``eta=2*pi*(y-y_bottom)/Ly``, use

    ``psi = A [sin(xi) sin(eta) + 0.35 cos(2 xi + 0.3) sin(eta)]``.

    The returned velocity is exactly ``u=-psi_y``, ``v=psi_x``.  During
    preprocessing the QG solver projects this field onto ``psi=0`` at the
    centered square and converts its relative vorticity to consistent PV.
    """
    length_x = x_right - x_left
    length_y = y_top - y_bottom
    wave_x = 2.0 * np.pi / length_x
    wave_y = 2.0 * np.pi / length_y
    xi = wave_x * (x - x_left)
    eta = wave_y * (y - y_bottom)

    x_structure = (
        np.sin(xi)
        + 0.35 * np.cos(2.0 * xi + 0.3)
    )
    u = -amplitude * wave_y * x_structure * np.cos(eta)
    v = amplitude * wave_x * (
        np.cos(xi)
        - 0.70 * np.sin(2.0 * xi + 0.3)
    ) * np.sin(eta)
    return u, v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        choices=(
            "manufactured_square_hole",
            "periodic_beta_square_hole",
            "channel_square_island",
            "periodic_square_cylinder",
            "basin_manufactured",
            "basin_northwest_manufactured",
            "basin_northwest_bump",
        ),
        default="manufactured_square_hole",
    )
    parser.add_argument("--resolution", type=int, default=128)
    arguments = parser.parse_args()
    case_name = arguments.case

    nx = ny = arguments.resolution
    exact_solution = None
    if case_name.startswith("basin_"):
        solver, exact_solution = make_basin_case(nx, case_name)
        animation_name = f"qg_{case_name}.gif"
    elif case_name == "manufactured_square_hole":
        x_left = y_bottom = -np.pi
        x_right = y_top = np.pi
        amplitude = deformation_radius = 1.0
        half_width = np.pi / 4.0
        mesh = grid(nx, ny, x_left, y_bottom, x_right, y_top)
        solver = qg(
            mesh,
            obstacle="centered",
            obstacle_size=(2.0 * half_width, 2.0 * half_width),
            deformation_radius=deformation_radius,
            beta=0.0,
            linear_drag=0.0,
        )

        def exact_velocity(x, y):
            return manufactured_square_hole_velocity(
                x,
                y,
                amplitude=amplitude,
                half_width=half_width,
            )

        def exact_pv(x, y):
            return manufactured_square_hole_pv(
                x,
                y,
                amplitude=amplitude,
                half_width=half_width,
                deformation_radius=deformation_radius,
            )

        solver.set_potential_vorticity_ivp(exact_pv, exact_velocity)
        solver.set_forcing_terms(
            lambda x, y: manufactured_square_hole_source(
                x,
                y,
                amplitude=amplitude,
                half_width=half_width,
            )
        )
        solver.cfl = 0.2
        solver.nu = 0.0
        solver.final_time = 2.0
        animation_name = "qg_manufactured_square_hole.gif"
        exact_solution = (
            lambda x, y, t: manufactured_square_hole_streamfunction(
                x,
                y,
                amplitude=amplitude,
                half_width=half_width,
            ),
            lambda x, y, t: exact_pv(x, y),
        )

    elif case_name == "periodic_beta_square_hole":
        parameters = {
            "amplitude": 1.0,
            "half_width": np.pi / 4.0,
            "beta": 1.0,
            "deformation_radius": 1.0,
        }
        mesh = grid(nx, ny, -np.pi, -np.pi, np.pi, np.pi)
        solver = qg(
            mesh,
            obstacle="centered",
            obstacle_size=(
                2.0 * parameters["half_width"],
                2.0 * parameters["half_width"],
            ),
            deformation_radius=parameters["deformation_radius"],
            beta=parameters["beta"],
        )
        solver.set_potential_vorticity_ivp(
            lambda x, y: manufactured_square_hole_pv(
                x,
                y,
                amplitude=parameters["amplitude"],
                half_width=parameters["half_width"],
                deformation_radius=parameters["deformation_radius"],
            ),
            lambda x, y: manufactured_square_hole_velocity(
                x,
                y,
                amplitude=parameters["amplitude"],
                half_width=parameters["half_width"],
            ),
        )
        solver.set_forcing_terms(
            lambda x, y: periodic_beta_square_hole_source(
                x,
                y,
                amplitude=parameters["amplitude"],
                half_width=parameters["half_width"],
                beta=parameters["beta"],
            )
        )
        solver.cfl = 0.2
        solver.nu = 0.0
        solver.final_time = 2.0
        animation_name = "qg_periodic_beta_square_hole.gif"
        exact_solution = (
            lambda x, y, t: manufactured_square_hole_streamfunction(
                x,
                y,
                amplitude=parameters["amplitude"],
                half_width=parameters["half_width"],
            ),
            lambda x, y, t: manufactured_square_hole_pv(
                x,
                y,
                amplitude=parameters["amplitude"],
                half_width=parameters["half_width"],
                deformation_radius=parameters["deformation_radius"],
            ),
        )

    elif case_name == "channel_square_island":
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
        mesh = grid(
            nx,
            ny,
            0.0,
            -0.5 * parameters["length_y"],
            parameters["length_x"],
            0.5 * parameters["length_y"],
        )
        solver = qg(
            mesh,
            obstacle="centered",
            obstacle_size=(
                2.0 * parameters["island_half_width"],
                2.0 * parameters["island_half_height"],
            ),
            deformation_radius=parameters["deformation_radius"],
            beta=parameters["beta"],
        )
        stationary_slip = (1.0, 0.0, 0.0)
        solver.domain.boundary_conditions = {
            "Channel": (None, stationary_slip, None, stationary_slip)
        }
        solver.set_potential_vorticity_ivp(
            lambda x, y: channel_square_island_pv(x, y, **parameters),
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
        solver.cfl = 0.2
        solver.nu = 0.0
        solver.final_time = 2.0
        animation_name = "qg_channel_square_island.gif"
        exact_solution = (
            lambda x, y, t: channel_square_island_streamfunction(
                x, y, **parameters
            ),
            lambda x, y, t: channel_square_island_pv(x, y, **parameters),
        )

    elif case_name == "periodic_square_cylinder":
        x_left = y_bottom = 0.0
        x_right = y_top = 2.0 * np.pi
        square_width = 0.5 * np.pi
        mesh = grid(nx, ny, x_left, y_bottom, x_right, y_top)
        solver = qg(
            mesh,
            obstacle="centered",
            obstacle_size=(square_width, square_width),
            deformation_radius=1.0,
            beta=0.0,
            linear_drag=0.01,
        )
        solver.set_ivp(
            lambda x, y: periodic_square_cylinder_vortex(
                x,
                y,
                x_left=x_left,
                x_right=x_right,
                y_bottom=y_bottom,
                y_top=y_top,
                amplitude=0.35,
            ),
            visual=0,
        )
        solver.cfl = 0.2
        solver.nu = 1.0e-3
        solver.final_time = 40.0
        animation_name = "qg_periodic_square_cylinder.gif"
    else:
        raise ValueError(f"Unknown QG tutorial case: {case_name}.")

    solver.update_pathname()
    solver.solve(save_freq=50, visual=0)
    solver.save_animation(
        ani_name=animation_name,
        del_data=True,
    )

    if exact_solution is not None:
        x = solver.X[:-1, :-1]
        y = solver.Y[:-1, :-1]
        boundary_mode = next(iter(solver.domain.boundary_conditions))
        fluid = solver._active_mask(boundary_mode)
        psi_exact = exact_solution[0](x, y, solver.t)
        q_exact = exact_solution[1](x, y, solver.t)
        print(
            f"{case_name} relative L2 errors: "
            f"psi={relative_l2_error(solver.psi_c, psi_exact, fluid):.6e}, "
            f"q={relative_l2_error(solver.potential_vorticity, q_exact, fluid):.6e}"
        )
        print(
            f"{case_name} Linf errors: "
            f"psi={np.max(np.abs((solver.psi_c - psi_exact)[fluid])):.6e}, "
            f"q={np.max(np.abs((solver.potential_vorticity - q_exact)[fluid])):.6e}"
        )
