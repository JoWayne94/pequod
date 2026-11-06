"""Two-layer QG manufactured benchmarks and a northwest-bump flow example.

Run from the repository root with PYTHONPATH=src. --convergence runs
32,64,128 grids and reports measured per-layer errors and orders.
"""

import argparse
from functools import partial
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pequod import grid, qg2
qg2 = partial(qg2, units="dimensionless")
from tutorials.incns.main import square_cylinder_wake
from tutorials.qg.main import (
    manufactured_square_hole_streamfunction,
    manufactured_square_hole_pv,
    manufactured_square_hole_velocity,
    manufactured_square_hole_source,
    channel_square_island_streamfunction,
    channel_square_island_pv,
    channel_square_island_velocity,
    channel_square_island_source,
    basin_manufactured_fields,
)


CASES = ("periodic_square_hole", "channel_square_island", "channel_square_island_manufactured",
         "basin_manufactured", "basin_northwest_manufactured",
         "basin_northwest_bump", "basin_northwest_bump_wind")


def northwest_bump_wind_source(x, y, amplitude=.1):
    """Upper-layer wind-stress curl on the unit basin; lower layer unforced.

    tau_x/(rho0 H1) = -amplitude*cos(2*pi*y)/(2*pi), tau_y = 0.
    Thus S1 = -d_y(tau_x/(rho0 H1)) = -amplitude*sin(2*pi*y).
    Negative southern curl drives the subtropical anticyclonic gyre.
    All quantities here are nondimensional; this is not an Atlantic hindcast.
    """
    x, y = np.broadcast_arrays(x, y)
    return np.stack([-amplitude*np.sin(2*np.pi*y), np.zeros_like(x)])


def base_fields(x, y, case):
    """Return phi, lap(phi), u, v, J(phi,lap(phi)) analytically."""
    if case == "periodic_square_hole":
        psi = manufactured_square_hole_streamfunction(x, y)
        q = manufactured_square_hole_pv(x, y)
        u, v = manufactured_square_hole_velocity(x, y)
        source = manufactured_square_hole_source(x, y)
    elif case == "channel_square_island_manufactured":
        psi = channel_square_island_streamfunction(x, y)
        q = channel_square_island_pv(x, y)
        u, v = channel_square_island_velocity(x, y)
        source = channel_square_island_source(x, y, beta=0.)
    else:
        psi, q, u, v, source = basin_manufactured_fields(
            x, y, bump=case != "basin_manufactured", amplitude=.01, beta=0.)
    return psi, q+psi, u, v, source


def exact_fields(x, y, case, F1=1., F2=2., beta=1.):
    """psi_i=a_i phi, q_i=a_i lap(phi)+F_i(a_j-a_i)phi.

    a=(1,1/2). Proportional psi makes J(psi_i,q_i)=a_i^2 J(phi,lap(phi)).
    The external source is this Jacobian plus beta*v_i.
    """
    phi, lap, u, v, jacobian = base_fields(x, y, case)
    weights = np.array([1., .5]).reshape((2,)+(1,)*np.ndim(phi))
    coupling = np.array([F1, F2]).reshape((2,)+(1,)*np.ndim(phi))
    psi = weights*phi
    pv = weights*lap + coupling*(weights[::-1]-weights)*phi
    return psi, pv, weights*u, weights*v, weights**2*jacobian+beta*weights*v


def make_case(n=64, case="periodic_square_hole", *, deformation_radius=None,
              wind_amplitude=.1, pv_diffusivity=5.e-4, bump_cells=None):
    if case not in CASES:
        raise ValueError(f"Unknown QG2 case: {case}")
    if case == "channel_square_island":
        # n is the cross-channel resolution: equal dx=dy on the 16 x 8 box.
        if n < 32 or n % 16:
            raise ValueError("Channel wake resolution must be >=32 and divisible by 16.")
        mesh = grid(2*n, n, 0., 0., 16., 8.)
        mesh.boundary_conditions = {"Channel": (None, (1., 0., 0.), None, (1., 0., 0.))}
        solver = qg2(mesh, obstacle="centered", obstacle_size=(1., 1.),
                     F1=.2, F2=.05, beta=0., nu=.01,
                     channel_throughflow=(1., .8), no_slip=True)
        profiles = [square_cylinder_wake(0., 16., 0., 8., bulk_velocity=U,
                                        perturbation=1.e-3) for U in (1., .8)]
        def velocity(x, y):
            fields = [profile(x, y) for profile in profiles]
            return np.stack([f[0] for f in fields]), np.stack([f[1] for f in fields])
        solver.set_ivp(velocity)
        solver.cfl = .15
        solver.final_time = 120.
        return solver, None
    if case == "periodic_square_hole":
        mesh = grid(n, n, -np.pi, -np.pi, np.pi, np.pi)
        obstacle, size = "centered", (np.pi/2, np.pi/2)
    elif case == "channel_square_island_manufactured":
        mesh = grid(n, n, 0., -np.pi/2, 2*np.pi, np.pi/2)
        mesh.boundary_conditions = {"Channel": (None, (1., 0., 0.), None, (1., 0., 0.))}
        obstacle, size = "centered", (np.pi/2, np.pi/2)
    else:
        mesh = grid(n, n, 0., 0., 1., 1.)
        mesh.boundary_conditions = {"Basin": ((1., 0., 0.),)*4}
        obstacle = False if case == "basin_manufactured" else "northwest_bump"
        size = (.25, .25)
    if case == "basin_northwest_bump_wind":
        radius = 1/np.sqrt(50.) if deformation_radius is None else deformation_radius
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("deformation_radius must be finite and positive.")
        if not np.isfinite(wind_amplitude) or wind_amplitude < 0:
            raise ValueError("wind_amplitude must be finite and nonnegative.")
        # Put the exposed southeast corner at (1/4, 1/2), aligned with
        # the wind-curl zero. This encourages, but does not impose, separation.
        size = (.25, .5)
        if bump_cells is not None:
            if not isinstance(bump_cells, (int, np.integer)) or not 0 < bump_cells < n:
                raise ValueError("bump_cells must be an integer strictly between 0 and n.")
            size = (bump_cells/n, bump_cells/n)
        # Positive beta selects western intensification. Resolved PV diffusion
        # regularizes the boundary current; drag acts directly only below.
        solver = qg2(mesh, obstacle=obstacle, obstacle_size=size,
                     F1=.8/radius**2, F2=.2/radius**2, beta=5., nu=pv_diffusivity,
                     linear_drag=(0., .05))
        zeros = lambda x, y: np.zeros((2,)+np.broadcast(x, y).shape)
        solver.set_potential_vorticity_ivp(zeros, lambda x, y: (zeros(x, y), zeros(x, y)))
        solver.set_forcing_terms(lambda x, y: northwest_bump_wind_source(x, y, wind_amplitude))
        solver.cfl = .15
        solver.final_time = 20.
        return solver, None
    dynamic = case == "basin_northwest_bump"
    solver = qg2(mesh, obstacle=obstacle, obstacle_size=size,
                 F1=1., F2=2., beta=1., nu=0.,
                 linear_drag=(0., .01) if dynamic else (0., 0.))
    fields = lambda x, y: exact_fields(x, y, case)
    solver.set_potential_vorticity_ivp(
        lambda x, y: fields(x, y)[1], lambda x, y: fields(x, y)[2:4])
    if not dynamic:
        solver.set_forcing_terms(lambda x, y: fields(x, y)[4])
    solver.cfl = .15
    solver.final_time = 2. if dynamic else .1
    return solver, None if dynamic else fields


def run_case(n, case, final_time=None, save_freq=5, make_gif=True):
    if make_gif and save_freq <= 0:
        raise ValueError("GIF generation requires save_freq > 0.")
    solver, exact = make_case(n, case)
    if final_time is not None:
        solver.final_time = final_time
    solver.data_dir = Path(__file__).parent / f"data_{case}_{n}"
    solver.solve(save_freq=save_freq, visual=-1)
    if not np.isfinite(solver.potential_vorticity).all():
        raise FloatingPointError("Two-layer PV became nonfinite.")
    errors = {}
    if exact:
        layer = solver.layers[0]
        psi, pv, *_ = exact(layer.X[:-1, :-1], layer.Y[:-1, :-1])
        mask = solver.active_mask
        for name, computed, reference in (("psi", solver.psi_c, psi),
                                          ("q", solver.potential_vorticity, pv)):
            for i in range(2):
                difference = (computed[i]-reference[i])[mask]
                errors[f"{name}{i+1}_l2"] = np.linalg.norm(difference)/np.linalg.norm(reference[i][mask])
                errors[f"{name}{i+1}_linf"] = np.max(abs(difference))
    print(f"{case}: N={n}, t={solver.t:g}, errors={errors}")
    if make_gif:
        output = solver.save_animation(
            Path(__file__).parent / f"qg2_{case}_{n}.gif")
        print(f"Saved GIF: {output}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=CASES, default=CASES[0])
    parser.add_argument("--resolution", type=int,
                        help="Grid size (default 64); wake uses Ny (default 128), Nx=2*Ny")
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--save-freq", type=int, default=5,
                        help="Save frames every N steps (default: 5); 0 disables saving with --no-gif")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip the four-panel GIF (convergence runs always skip it)")
    parser.add_argument("--convergence", action="store_true")
    args = parser.parse_args()
    if args.resolution is None:
        args.resolution = 128 if args.case == "channel_square_island" else 64
    if args.save_freq < 0 or (args.save_freq == 0 and not (args.no_gif or args.convergence)):
        parser.error("--save-freq must be positive for GIF output, or nonnegative with --no-gif/--convergence.")
    if args.convergence and args.case in {"basin_northwest_bump", "basin_northwest_bump_wind", "channel_square_island"}:
        parser.error("Dynamic examples have no exact reference for error convergence.")
    previous = None
    for n in ((32, 64, 128) if args.convergence else (args.resolution,)):
        errors = run_case(n, args.case, args.final_time, args.save_freq,
                          make_gif=not (args.no_gif or args.convergence))
        if previous:
            print("Observed orders:", {key: np.log2(previous[key]/value)
                                       for key, value in errors.items()})
        previous = errors
