"""Wind-driven QG2 rectangular basin without an obstacle.

Same SI inputs, wind stress tau_x=W*sin(pi*y/L), Navier-slip walls,
diagnostics and four-panel GIF as wind_experiment.py. Final time is in
Julian years; interval in seconds. Both PV modes use plain DST inverses:
there is no obstacle capacitance matrix or Green-function correction.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tutorials.qg2 import wind_experiment as _shared
from pequod.solvers.qg_units import DEFAULT_BETA, DEFAULT_DRAG, DEFAULT_NU

wind_source_amplitude = _shared.wind_source_amplitude


def make_wind_solver(n, radius, wind, nu=DEFAULT_NU, beta=DEFAULT_BETA,
                     drag=DEFAULT_DRAG, *, slip_length=30000., cfl=.15):
    """Construct the same SI experiment, with four straight regular walls."""
    return _shared.make_wind_solver(n, radius, wind, nu, beta, drag,
                                    slip_length=slip_length, cfl=cfl, obstacle=False)


def run(n=512, radius=38400., wind=None, nu=DEFAULT_NU, final_time=20., interval=864000.,
        output=None, gif=True, *, beta=DEFAULT_BETA, drag=DEFAULT_DRAG,
        wind_stress=None, density=1025., upper_depth=500., slip_length=30000., cfl=.15):
    """Run and save snapshots/diagnostics/GIF in a distinct wind2_SI directory."""
    return _shared.run(n, radius, wind, nu, final_time, interval, output, gif,
                       beta=beta, drag=drag, wind_stress=wind_stress, density=density,
                       upper_depth=upper_depth, slip_length=slip_length, cfl=cfl, obstacle=False)


if __name__ == "__main__":
    _shared.main(obstacle=False, description=__doc__)
