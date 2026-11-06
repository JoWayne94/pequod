"""Wind-driven QG2 basin: SI inputs, --final-time in Julian years.

Radius is in metres, interval in seconds, wind stress in N/m^2, and an
optional direct --wind PV-source amplitude in s^-2. A year is 365.25 days.
The 3840 km basin and reference velocity 0.01 m/s define the model scales.
Use --bump-cells K for a square, or supply both --bump-width-cells NX and
--bump-height-cells NY for an independently sized northwest rectangle.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pequod import grid, qg2
from pequod.solvers import _qg_kernels
from pequod.solvers.qg_units import (
    QG_SCALES, SECONDS_PER_YEAR, DEFAULT_BETA, DEFAULT_DRAG, DEFAULT_NU,
)


def wind_source_amplitude(wind=None, wind_stress=None, density=1025., upper_depth=500.):
    """Return the sinusoidal PV-source amplitude in s^-2.

    tau_x=W*sin(pi*y/L), tau_y=0, with W given by --wind-stress.
    Thus S1=-pi*W*cos(pi*y/L)/(rho0*H1*L). Only layer 1 is forced.
    The direct --wind option instead supplies the positive source amplitude.
    """
    if wind is not None and wind_stress is not None:
        raise ValueError("Specify either wind (PV source) or wind_stress, not both.")
    if not all(np.isfinite(v) and v > 0 for v in (density, upper_depth)):
        raise ValueError("density and upper_depth must be finite and positive.")
    stress = .1 if wind_stress is None else wind_stress
    amplitude = np.pi*stress/(density*upper_depth*QG_SCALES.length) if wind is None else wind
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Wind amplitude must be finite and nonnegative.")
    return float(amplitude)


def wind_obstacle_size(n, bump_cells=None, *, bump_width_cells=None,
                       bump_height_cells=None, obstacle=True):
    """Resolve (width, height) in metres; reject ambiguous/incomplete inputs."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0:
        raise ValueError("Resolution must be a positive integer.")
    rectangular = bump_width_cells is not None or bump_height_cells is not None
    if bump_cells is not None and rectangular:
        raise ValueError("Use either bump_cells or both bump_width_cells and bump_height_cells, not both.")
    if not obstacle:
        if bump_cells is not None or rectangular:
            raise ValueError("Bump dimensions require the northwest obstacle.")
        return None
    if rectangular and (bump_width_cells is None or bump_height_cells is None):
        raise ValueError("Specify both bump_width_cells and bump_height_cells.")
    for name, value in (("bump_cells", bump_cells), ("bump_width_cells", bump_width_cells),
                         ("bump_height_cells", bump_height_cells)):
        if value is not None and (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or not 0 < value < n):
            raise ValueError(f"{name} must be an integer strictly between 0 and resolution.")
    L = QG_SCALES.length
    if rectangular:
        return bump_width_cells*L/n, bump_height_cells*L/n
    return (.25*L, .5*L) if bump_cells is None else (bump_cells*L/n,)*2


def make_wind_solver(n, radius, wind, nu=DEFAULT_NU, beta=DEFAULT_BETA,
                     drag=DEFAULT_DRAG, bump_cells=None, *, slip_length=30000., cfl=.15,
                     obstacle=True, bump_width_cells=None, bump_height_cells=None):
    """SI-configured wind basin. Geometry in m, PV source in s^-2."""
    L = QG_SCALES.length
    size = wind_obstacle_size(n, bump_cells, bump_width_cells=bump_width_cells,
                              bump_height_cells=bump_height_cells, obstacle=obstacle)
    mesh = grid(n, n, 0., 0., L, L)
    mesh.boundary_conditions = {"Basin": ((1., 0., 0.),)*4}
    solver = qg2(mesh, obstacle="northwest_bump" if obstacle else False, obstacle_size=size,
                 F1=.8/radius**2, F2=.2/radius**2, beta=beta, nu=nu,
                 linear_drag=(0., drag), slip_length=slip_length)
    zeros = lambda x, y: np.zeros((2,)+np.broadcast(x, y).shape)
    solver.set_potential_vorticity_ivp(zeros, lambda x, y: (zeros(x, y), zeros(x, y)))
    solver.set_forcing_terms(lambda x, y: np.stack([-wind*np.cos(np.pi*y/L), np.zeros_like(x)]))
    solver.cfl = cfl
    return solver


def run(n=512, radius=38400., wind=None, nu=DEFAULT_NU, final_time=20., interval=864000.,
        output=None, gif=True, bump_cells=None, *, beta=DEFAULT_BETA,
        drag=DEFAULT_DRAG, wind_stress=None, density=1025., upper_depth=500.,
        slip_length=30000., cfl=.15, obstacle=True, bump_width_cells=None, bump_height_cells=None):
    L, t0 = QG_SCALES.length, QG_SCALES.time
    if n <= 0 or n % 4 or not np.isfinite(radius) or radius*n/L < 1:
        raise ValueError("Require positive N divisible by 4 and N*Ld/L >= 1 (Ld in metres).")
    if not np.isfinite(final_time) or final_time <= 0 or not np.isfinite(interval) or interval <= 0:
        raise ValueError("final_time and interval must be finite and positive.")
    size = wind_obstacle_size(n, bump_cells, bump_width_cells=bump_width_cells,
                              bump_height_cells=bump_height_cells, obstacle=obstacle)
    source = wind_source_amplitude(wind, wind_stress, density, upper_depth)
    solver = make_wind_solver(n, radius, source, nu, beta, drag, bump_cells,
                              slip_length=slip_length, cfl=cfl, obstacle=obstacle,
                              bump_width_cells=bump_width_cells, bump_height_cells=bump_height_cells)
    engine = solver.dimensionless
    final_seconds = final_time*SECONDS_PER_YEAR
    output = Path(output) if output else Path(__file__).parent / (
        f"{'wind' if obstacle else 'wind2'}_SI_N{n}_Ld{radius:g}_W{source:g}_nu{nu:g}_{time.time_ns()}")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(format_version=3, input_units="SI", frame_units="dimensionless",
                  wind_stress_profile="tau_x=W*sin(pi*y/L), tau_y=0",
                  pv_source_profile="S1=-amplitude*cos(pi*y/L), S2=0",
                  transport_backend="numba" if _qg_kernels.AVAILABLE else "numpy",
                  n=n, deformation_radius=radius, wind=source, nu=nu,
                  slip_length_m=slip_length, cfl=cfl,
                  lateral_boundary_condition="Navier slip" if slip_length is not None else "legacy mask",
                  wind_stress=(.1 if wind_stress is None else wind_stress) if wind is None else None,
                  density=density, upper_depth=upper_depth,
                  F1=.8/radius**2, F2=.2/radius**2, beta=beta, drag=[0., drag],
                  reference_length_m=L, reference_velocity_m_s=QG_SCALES.velocity,
                  reference_time_s=t0, seconds_per_year=SECONDS_PER_YEAR,
                  final_time_years=final_time, final_time_seconds=final_seconds, interval_seconds=interval,
                  domain_over_radius=L/radius, points_per_radius=n*radius/L,
                  grid_spacing_m=L/n,
                  obstacle="northwest_bump" if obstacle else False,
                  bump_cells=None if bump_cells is None else int(bump_cells),
                  bump_width_cells=int(round(size[0]*n/L)) if size else None,
                  bump_height_cells=int(round(size[1]*n/L)) if size else None,
                  obstacle_size=list(size) if size else None,
                  dimensionless=dict(beta=engine.beta, nu=list(engine.nu), drag=list(engine.linear_drag),
                                     F1=engine.coupling[0], F2=engine.coupling[1],
                                     wind=source*t0**2, radius=radius/L,
                                     slip_length=None if slip_length is None else slip_length/L,
                                     final_time=final_seconds/t0))
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(json.dumps(config), flush=True)
    print(f"Output: {output.resolve()}", flush=True)
    started = time.monotonic()
    solver.initialize()
    print(f"Initialized in {time.monotonic()-started:.1f}s", flush=True)
    print(f"Transport: {config['transport_backend']}; PV inverses: "
          f"{engine._barotropic_solver.backend}, {engine._baroclinic_solver.backend}", flush=True)
    if not _qg_kernels.AVAILABLE:
        print("For the compiled fast path, install project dependencies with uv sync "
              "and unset PEQUOD_QG_NUMBA=0.", flush=True)
    frames = []
    step = 0
    last_report = time.monotonic()
    with (output / "diagnostics.jsonl").open("w") as diagnostics:
        def save():
            q, psi = engine.potential_vorticity, engine.psi_c
            if not np.isfinite(q).all() or not np.isfinite(psi).all():
                raise FloatingPointError("Nonfinite QG state.")
            path = output / f"frame_{len(frames):05d}.npz"
            vmax = solver.max_velocity_by_layer_m_s
            np.savez_compressed(path, t=engine.t, t_seconds=solver.t,
                                t_years=solver.t/SECONDS_PER_YEAR, q=q, psi=psi,
                                max_velocity_m_s=vmax)
            frames.append(path)
            record = dict(t_dimensionless=engine.t, t_seconds=solver.t, t_years=solver.t/SECONDS_PER_YEAR,
                          dt_seconds=solver.dt, max_velocity_m_s=float(np.max(vmax)),
                          layer_max_velocity_m_s=vmax.tolist(),
                          step=step, wall_seconds=time.monotonic()-started,
                          psi_max_m2_s=(np.max(abs(psi), axis=(1, 2))*QG_SCALES.streamfunction).tolist(),
                          pv_rms_s_1=(np.sqrt(np.mean(q[:, engine.active_mask]**2, axis=1))/t0).tolist())
            diagnostics.write(json.dumps(record)+"\n")
            diagnostics.flush()
            print(json.dumps(record), flush=True)
            (output / "frames.json").write_text(json.dumps([p.name for p in frames]))
        save()
        while engine.t < final_seconds/t0:
            solver.final_time = min(final_seconds, len(frames)*interval)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                while engine.t < engine.final_time:
                    solver.step()
                    step += 1
                    if time.monotonic()-last_report > 30:
                        vmax = solver.max_velocity_by_layer_m_s
                        print(f"Progress t={solver.t/SECONDS_PER_YEAR:.6g} years, steps={step}, "
                              f"dt={solver.dt:.3g} s, max velocity={np.max(vmax):.6g} m/s "
                              f"(layers {vmax.tolist()} m/s)", flush=True)
                        last_report = time.monotonic()
            save()
    if gif:
        engine._saved_frames = frames
        solver.save_animation(output / "qg2.gif")
    print(f"Completed: {output.resolve()}", flush=True)
    return output


def main(*, obstacle=True, description=None):
    """Shared CLI for the bumped and rectangular-basin experiments."""
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--radius", type=float, default=38400., help="Baroclinic deformation radius (m)")
    forcing = parser.add_mutually_exclusive_group()
    forcing.add_argument("--wind", type=float, help="Direct PV-source amplitude W (s^-2)")
    forcing.add_argument("--wind-stress", type=float,
                         help="W in tau_x=W*sin(pi*y/L) (N/m^2); default 0.1")
    parser.add_argument("--density", type=float, default=1025., help="Reference density (kg/m^3)")
    parser.add_argument("--upper-depth", type=float, default=500., help="Upper-layer depth used for wind curl (m)")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Planetary gradient (m^-1 s^-1)")
    parser.add_argument("--drag", type=float, default=DEFAULT_DRAG, help="Lower-layer drag (s^-1)")
    parser.add_argument("--nu", type=float, default=DEFAULT_NU, help="PV diffusivity (m^2/s)")
    parser.add_argument("--cfl", type=float, default=.15, help="Advective CFL in (0,1]; default 0.15")
    parser.add_argument("--slip-length", type=float, default=30000.,
                        help="Navier slip length on all walls (m); default 30000; 0=no slip, inf=free slip")
    parser.add_argument("--final-time", type=float, default=20., help="Final time (years of 365.25 days)")
    parser.add_argument("--interval", type=float, default=864000., help="Snapshot interval (seconds); default 10 days")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-gif", action="store_true")
    if obstacle:
        parser.add_argument("--bump-cells", type=int,
                            help="Northwest square bump width/height in grid spacings (default: original rectangle)")
        parser.add_argument("--bump-width-cells", type=int,
                            help="Northwest bump width in x grid spacings; requires --bump-height-cells")
        parser.add_argument("--bump-height-cells", type=int,
                            help="Northwest bump height in y grid spacings; requires --bump-width-cells")
    else:
        parser.set_defaults(bump_cells=None, bump_width_cells=None, bump_height_cells=None)
    args = parser.parse_args()
    try:
        wind_obstacle_size(args.resolution, args.bump_cells, bump_width_cells=args.bump_width_cells,
                           bump_height_cells=args.bump_height_cells, obstacle=obstacle)
    except ValueError as error:
        parser.error(str(error))
    run(args.resolution, args.radius, args.wind, args.nu, args.final_time,
        args.interval, args.output, not args.no_gif, args.bump_cells,
        beta=args.beta, drag=args.drag, wind_stress=args.wind_stress,
        density=args.density, upper_depth=args.upper_depth,
        slip_length=args.slip_length, cfl=args.cfl, obstacle=obstacle,
        bump_width_cells=args.bump_width_cells, bump_height_cells=args.bump_height_cells)


if __name__ == "__main__":
    main()
