"""Matched-physics QG2 speed/equivalence benchmark, excluding one-time JIT.

PYTHONPATH=src .venv/bin/python tutorials/qg2/benchmark.py --steps 5000
The baseline uses the original CABARET stencils and sparse inverses.
Both paths use identical forcing, wall physics, CFL and source timestep cap.
"""
import argparse
import cProfile
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pequod.gulf import Gulf
from pequod.solvers import _qg_kernels
from pequod.solvers.incns import IncNavierStokes
from pequod.solvers.qg import _MaskedHelmholtzSolver
from tutorials.qg2.wind_experiment import make_wind_solver, wind_source_amplitude


def benchmark(n=128, steps=5000, radius=40000., slip_length=120000., profile_dir=None):
    available = _qg_kernels.AVAILABLE
    results, states = {}, []
    try:
        for label in ("baseline", "optimized"):
            _qg_kernels.AVAILABLE = available and label == "optimized"
            s = make_wind_solver(n, radius, wind_source_amplitude(),
                                 bump_cells=4, slip_length=slip_length, cfl=.4)
            s.initialize()
            e = s.dimensionless
            if label == "baseline":
                kwargs = dict(periodic_x=False, periodic_y=False, backend="sparse")
                e._barotropic_solver = _MaskedHelmholtzSolver(e.active_mask,
                    e.domain.dx, e.domain.dy, 0., **kwargs)
                e._baroclinic_solver = _MaskedHelmholtzSolver(e.active_mask,
                    e.domain.dx, e.domain.dy, sum(e.coupling), **kwargs)
                for layer in e.layers:
                    layer.predictor = Gulf.predictor_corrector(layer,
                        lambda layer=layer: Gulf._predictor(layer))
                    layer.corrector = Gulf.predictor_corrector(layer, layer._corrector)
                    for name in ("second_order_extrapolation", "numerical_flux_x", "numerical_flux_y"):
                        setattr(layer, name, MethodType(getattr(Gulf, name), layer))
                    # flux_funcs contains bound methods captured during initialization.
                    layer._average_psi_to_vertices = MethodType(IncNavierStokes._average_psi_to_vertices, layer)
                    layer._boundary_conditions()
                    layer._psi_boundary_conditions()
            e.final_time = 100.
            for _ in range(32):
                e.step()
            p = cProfile.Profile() if profile_dir else None
            if p:
                p.enable()
            started = time.perf_counter()
            for _ in range(steps):
                e.step()
            elapsed = time.perf_counter()-started
            if p:
                p.disable()
                Path(profile_dir).mkdir(parents=True, exist_ok=True)
                p.dump_stats(str(Path(profile_dir)/f"{label}.prof"))
            results[label] = dict(seconds=elapsed, milliseconds_per_step=1000*elapsed/steps,
                                  model_time=e.t, compiled=_qg_kernels.AVAILABLE,
                                  inversion=e._barotropic_solver.backend)
            states.append((e.potential_vorticity.copy(), e.psi_c.copy()))
        results["speedup"] = results["baseline"]["seconds"]/results["optimized"]["seconds"]
        for k, name in enumerate(("pv", "psi")):
            a, b = states[0][k], states[1][k]
            results[f"relative_{name}_difference"] = float(np.linalg.norm(a-b)/np.linalg.norm(a))
        print(json.dumps(results, indent=2))
        return results
    finally:
        _qg_kernels.AVAILABLE = available


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--radius", type=float, default=40000.)
    parser.add_argument("--slip-length", type=float, default=120000.)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    benchmark(args.resolution, args.steps, args.radius, args.slip_length, args.profile_dir)
