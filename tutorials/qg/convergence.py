"""Convergence study for the periodic QG square-hole manufactured solution.

The default resolution sequence is 64, 128, 256, and 512.  The largest
masked sparse factorization is intentionally part of the benchmark but can
require substantial memory; pass a shorter ``--resolutions`` list for a
quick development run.
"""

import argparse
from functools import partial
import csv
from pathlib import Path
import warnings

import numpy as np

from pequod import grid, qg
qg = partial(qg, units="dimensionless")
try:
    from .main import (
        manufactured_square_hole_pv,
        manufactured_square_hole_source,
        manufactured_square_hole_streamfunction,
        manufactured_square_hole_velocity,
    )
except ImportError:
    from main import (
        manufactured_square_hole_pv,
        manufactured_square_hole_source,
        manufactured_square_hole_streamfunction,
        manufactured_square_hole_velocity,
    )


def _relative_l2(numerical, exact, mask):
    difference = numerical[mask] - exact[mask]
    return np.linalg.norm(difference) / np.linalg.norm(exact[mask])


def _observed_order(previous_error, error, previous_n, n):
    if previous_error is None:
        return np.nan
    return np.log(previous_error / error) / np.log(n / previous_n)


def run_level(n, final_time=0.5, cfl=0.15):
    """Run one grid level and return the requested error diagnostics."""
    amplitude = deformation_radius = 1.0
    half_width = np.pi / 4.0
    mesh = grid(n, n, -np.pi, -np.pi, np.pi, np.pi)
    solver = qg(
        mesh,
        obstacle="centered",
        obstacle_size=(2.0 * half_width, 2.0 * half_width),
        deformation_radius=deformation_radius,
        beta=0.0,
        linear_drag=0.0,
    )

    def exact_pv(x, y):
        return manufactured_square_hole_pv(
            x,
            y,
            amplitude=amplitude,
            half_width=half_width,
            deformation_radius=deformation_radius,
        )

    def exact_velocity(x, y):
        return manufactured_square_hole_velocity(
            x,
            y,
            amplitude=amplitude,
            half_width=half_width,
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
    solver.cfl = cfl
    solver.nu = 0.0
    solver.final_time = final_time
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for preprocess in solver.get_pre_processes():
            preprocess()

        x = solver.X[:-1, :-1]
        y = solver.Y[:-1, :-1]
        fluid = ~solver.obstacle_mask
        psi_exact = manufactured_square_hole_streamfunction(
            x,
            y,
            amplitude=amplitude,
            half_width=half_width,
        )
        q_exact = exact_pv(x, y)
        inversion_psi_relative_l2 = _relative_l2(
            solver.psi_c,
            psi_exact,
            fluid,
        )

        # Apply the same five-point Helmholtz stencil to the analytic psi.
        # Solid values are zero because psi=0 on the square boundary and the
        # masked inverse does not include points inside the hole.
        psi_for_stencil = psi_exact.copy()
        psi_for_stencil[solver.obstacle_mask] = 0.0
        discrete_laplacian = (
            (
                np.roll(psi_for_stencil, -1, axis=1)
                - 2.0 * psi_for_stencil
                + np.roll(psi_for_stencil, 1, axis=1)
            )
            / mesh.dx**2
            + (
                np.roll(psi_for_stencil, -1, axis=0)
                - 2.0 * psi_for_stencil
                + np.roll(psi_for_stencil, 1, axis=0)
            )
            / mesh.dy**2
        )
        discrete_q = (
            discrete_laplacian
            - solver.inverse_deformation_radius_squared * psi_for_stencil
        )
        operator_q_relative_l2 = _relative_l2(
            discrete_q,
            q_exact,
            fluid,
        )

        tolerance = np.finfo(np.float64).eps
        while abs(solver.t - solver.final_time) > tolerance:
            for advance in solver.get_solve_sequence():
                advance()
            solver.t += solver.dt

    psi_difference = solver.psi_c - psi_exact
    q_difference = solver.potential_vorticity - q_exact
    flux_x, flux_y = solver.physical_flux()

    return {
        "n": n,
        "dx": mesh.dx,
        "final_time": solver.t,
        "inversion_psi_relative_l2": inversion_psi_relative_l2,
        "operator_q_relative_l2": operator_q_relative_l2,
        "psi_relative_l2": _relative_l2(solver.psi_c, psi_exact, fluid),
        "q_relative_l2": _relative_l2(
            solver.potential_vorticity,
            q_exact,
            fluid,
        ),
        "psi_linf": np.max(np.abs(psi_difference[fluid])),
        "q_linf": np.max(np.abs(q_difference[fluid])),
        "obstacle_normal_flux": max(
            np.max(np.abs(flux_x[:, solver.obstacle_faces_x])),
            np.max(np.abs(flux_y[:, solver.obstacle_faces_y])),
        ),
        "forcing_integral": (
            np.sum(solver.external_forcing[fluid]) * mesh.dx * mesh.dy
        ),
    }


def run_convergence(
    resolutions=(64, 128, 256, 512),
    *,
    final_time=0.5,
    cfl=0.15,
):
    """Run all levels and append observed L2 convergence orders."""
    results = []
    previous = None
    for n in resolutions:
        result = run_level(int(n), final_time=final_time, cfl=cfl)
        result["psi_order"] = _observed_order(
            None if previous is None else previous["psi_relative_l2"],
            result["psi_relative_l2"],
            None if previous is None else previous["n"],
            result["n"],
        )
        result["q_order"] = _observed_order(
            None if previous is None else previous["q_relative_l2"],
            result["q_relative_l2"],
            None if previous is None else previous["n"],
            result["n"],
        )
        result["inversion_psi_order"] = _observed_order(
            None
            if previous is None
            else previous["inversion_psi_relative_l2"],
            result["inversion_psi_relative_l2"],
            None if previous is None else previous["n"],
            result["n"],
        )
        result["operator_q_order"] = _observed_order(
            None
            if previous is None
            else previous["operator_q_relative_l2"],
            result["operator_q_relative_l2"],
            None if previous is None else previous["n"],
            result["n"],
        )
        results.append(result)
        previous = result
    return results


def _print_results(results):
    header = (
        "N      E_inv(psi)  p_inv    E_op(q)     p_op     "
        "E_t(psi)    p_psi    E_t(q)      p_q"
    )
    print(header)
    for result in results:
        print(
            f"{result['n']:<6d} "
            f"{result['inversion_psi_relative_l2']:<11.4e} "
            f"{result['inversion_psi_order']:<8.3f} "
            f"{result['operator_q_relative_l2']:<11.4e} "
            f"{result['operator_q_order']:<8.3f} "
            f"{result['psi_relative_l2']:<11.4e} "
            f"{result['psi_order']:<8.3f} "
            f"{result['q_relative_l2']:<11.4e} "
            f"{result['q_order']:<8.3f}"
        )


def _write_csv(results, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=(64, 128, 256, 512),
    )
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("qg_manufactured_convergence.csv"),
    )
    arguments = parser.parse_args()
    convergence = run_convergence(
        arguments.resolutions,
        final_time=arguments.final_time,
        cfl=arguments.cfl,
    )
    _print_results(convergence)
    _write_csv(convergence, arguments.output)
    print(f"Wrote {arguments.output}")
