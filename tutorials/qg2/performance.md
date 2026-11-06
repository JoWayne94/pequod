# QG wind experiment: wall conditions and performance

## Second optimization pass (16 October 2026)

The fresh 128² profile again identified three dominant groups:

1. Sparse triangular PV solves: about 41% of stepping time.
2. CABARET divergence, extrema, extrapolation and upwinding: about 25--30%.
3. Velocity reconstruction and its limiter-source increments: about 10--11%.

The basin inverse now automatically uses DST-I transforms plus an exact small
capacitance correction for the solid. This inverts the **same five-point
operator**, including zero Dirichlet data on exposed obstacle nodes. It is not a
spectral change of discretization. Tests compare random right-hand sides and
discrete residuals with sparse LU, including rectangular grids, Poisson and
Helmholtz modes, centered holes and both corner bumps. Large obstacles retain
the sparse backend: the fast path caps the exposed constraint count at 64 and
the stored Green functions at 16 MiB per mode. Periodic/channel inverses retain
their existing backends.

Transport, vertex averaging, diffusion and velocity/limiter loops now have
cached Numba kernels with float64 arithmetic and **no fast-math**. Numba is a
project dependency; `uv sync` installs it. The equivalent reusable-buffer NumPy
path remains available with `PEQUOD_QG_NUMBA=0`. The wind runner prints both
selected backends. Initial JIT compilation is excluded from warm benchmarks.
Numba's compilation and caching options are documented in the
[Numba reference](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html).

An initial 5,000-step matched-physics run measured 17.22 s before versus 10.00 s
after, a **1.72× speedup**. Both advanced to identical model times; relative PV
and streamfunction differences were respectively 2.4e-13 and 3.9e-14. This
projects a 90-minute integration to about **52 minutes**, assuming comparable
step counts and machine load. It is not a completed century-run measurement.
The changed stress profile can change currents and the adaptive step count;
GIF rendering and output add time. No sub-hour wall-clock guarantee is made.
A stepping-only one-year pilot with the new wind, N=128, radius 40 km,
four-cell bump, slip length 120 km and CFL 0.4 completed 16,166 steps in
33.9 seconds, with finite fields and maximum speed 0.98 m/s. This tests spin-up,
not century-long equilibrium or long-run step counts. A short end-to-end CLI
equivalent also produced the four-panel GIF successfully.

Reproduce a matched comparison (SI parameters; the defaults match the saved
128² run with radius 40 km and slip length 120 km):

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python \
  tutorials/qg2/benchmark.py --steps 5000
```

Add `--profile-dir /tmp/qg-profile` to write separate cProfile samples. The
baseline intentionally uses the original CABARET stencils and sparse inverses;
forcing, slip physics and timestep restrictions are identical on both paths.
The explicit beta cap is **unchanged**. A trial using a looser basin-frequency
bound failed the wind time-accuracy comparison and was discarded.

The SI runner now uses `tau_x=W*sin(pi*y/L)`, `tau_y=0`, where `W` is
`--wind-stress` in N/m², on `0 <= y <= L`. Consequently:

```
S1 = -pi*W/(rho0*H1*L) * cos(pi*y/L),   S2 = 0.
```

For W=0.1, rho0=1025, H1=500 and L=3,840,000 in SI, the source amplitude is
1.59633773e-13 s^-2. The direct `--wind` option supplies that source amplitude,
not stress. Format-version-3 configurations record the profile explicitly.
The older dimensionless wind benchmark in `main.py` remains unchanged.

## Obstacle-free companion

`wind_experiment2.py` shares the wind runner and all SI options except
the bump-size options, but uses the full rectangular basin. Both modal inverses print
`dst`: no capacitance matrix, cached obstacle Green functions, or obstacle
correction is constructed or used. It retains the same Navier-slip outer
walls, rest IVP, source profile, diagnostics and four-panel GIF. Output folders
start with `wind2_SI_` and record `obstacle=false`, `obstacle_size=null`.
Removing the obstacle changes the flow as well as removing its numerical cost;
it is not a matched-geometry performance comparison.
A short 128² warm-step sample (radius 40 km, slip length 120 km, CFL 0.4)
measured roughly 1.70 ms/step with the four-cell bump versus 1.53 ms/step
without it: about 10% lower per-step cost. Adaptive step counts may differ
over a full run, so this is not a whole-run speedup guarantee.

The SI source is converted exactly once by
`QGSIInterface.set_forcing_terms`: `S_dimensionless = S_SI*T0**2`, with
`T0=384000000` seconds. Neither runner should pre-scale its callback. A test
also checks the first physical PV increment from rest, `q1 = dt_seconds*S1`,
to detect missing or duplicate scaling independently of the stored source.

## Independent bump width and height

`wind_experiment.py` accepts `--bump-width-cells NX --bump-height-cells NY`.
Width is the x extent measured eastward from the west wall; height is the y
extent measured southward from the north wall. Both values must be integers
strictly between zero and the resolution. Supply both flags together and do
not mix them with `--bump-cells K`, which remains the square-bump shorthand.
With no size flags, the original width 0.25L and height 0.5L are unchanged.

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -u tutorials/qg2/wind_experiment.py \
  --resolution 128 --radius 40000 \
  --bump-width-cells 4 --bump-height-cells 8 \
  --wind-stress 0.1 --beta 2e-11 --drag 4e-8 --nu 100 \
  --slip-length 120000 --cfl 0.4 --final-time 100 --interval 8640000
```

This is a 120 km wide by 240 km tall NW bump on the 30 km grid. The config
records resolved `bump_width_cells`, `bump_height_cells` and the authoritative
physical `obstacle_size=[width_m,height_m]`. Energy/mean-flow postprocessing
already reads the latter, so rectangular masks require no separate plotting
parameters. The obstacle-free companion does not accept bump dimensions.

## Previous optimization pass (historical measurements)

The completed 128², 180 km deformation-radius experiment simulated **100 years**.
Its diagnostics record 1,899,257 integration steps and 10,170 seconds (about 2.8
hours) before animation rendering. Changing output frequency alone would not
remove this integration cost.

## Measured bottlenecks and changes

A cProfile sample of 1,200 warm steps used a 128² basin, 192 km deformation radius,
four-spacing northwest bump, default SI coefficients and wind, CFL 0.15, and the
legacy wall mask. Before/after physics and timesteps were matched. Times include
profiling overhead and vary with the machine and load.

| Bottleneck | Before | After | Change |
|---|---:|---:|---|
| Sparse triangular solves | 3.56 s (54% of total) | 1.91 s | Symmetric minimum-degree ordering reduces factor fill; reuse modal RHS/solution arrays. |
| Velocity reconstruction | 0.56 s (8%) | 0.50 s | Reuse raw-velocity history and gradient storage; compute each common limiter increment once. |
| Legacy wall masking | 0.48 s (7%) | 0.022 s | Cache flat solid/face indices instead of repeatedly scanning full boolean grids. |

Total stepping time decreased from **6.65 s to 4.47 s**, about **1.5× faster**.
With the new 30 km partial-slip condition, the same-sized sample took **4.53 s**.
Partial-slip wall values are recomputed after inversion and cached for flux
restoration. New boundary physics changes the solution, so a full-run duration
cannot be inferred by scaling the old duration.

The sparse ordering uses SciPy's documented
[`MMD_AT_PLUS_A`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)
option. Both modal inversions still occur at both CABARET stages; no transport
step or physical forcing has been omitted.

## CFL and the number of steps

`--cfl 0.4` raises the advection and diffusion Courant limits. The explicit beta
and drag source has a separate cap using `min(cfl, 0.15)`, retaining the previous
source restriction when advection CFL increases. With the default SI beta/drag
and 3840 km basin, this cap is about 1,952 seconds. A century therefore still
requires at least roughly 1.6 million steps; advection, diffusion, and output-time
clipping can require more. The ratio 0.4/0.15 is not a whole-run speedup prediction.

## Wall physics

The wind runner now defaults to `--slip-length 30000` (metres) on every outer
wall and exposed island/bump face. Stationary Navier slip is `u_t = ell*d_n u_t`,
with normal `n` into fluid, together with impermeability. Zero length is no slip;
`inf` is free slip. Generic qg/qg2 constructors accept the same SI length. `None`
retains the old benchmark mask, used by the unchanged manufactured cases.

Boundary PV includes the stretching term, not just relative vorticity. Corners
average adjoining face curvatures. Omitted north/east boundary nodes enter the
PV diffusion stencil directly. These closures are for stationary, straight,
grid-aligned faces and do not assert curved-boundary accuracy.

## Reproduce a kernel profile

Run from the repository root. Set `slip_length=None` for the matched legacy-mask
case, or use `30000.` for the new wind wall condition.

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python - <<'PY'
import cProfile
import pstats
from tutorials.qg2.wind_experiment import make_wind_solver, wind_source_amplitude

solver = make_wind_solver(
    128, 192000., wind_source_amplitude(),
    bump_cells=4, slip_length=30000., cfl=.4,
)
solver.initialize()
engine = solver.dimensionless
engine.final_time = 100.  # Model time; prevents final-time clipping in this sample.
profile = cProfile.Profile()
profile.enable()
for _ in range(1200):
    engine.step()
profile.disable()
pstats.Stats(profile).strip_dirs().sort_stats('cumtime').print_stats(20)
PY
```

Correctness checks cover finite/zero/infinite slip lengths, all regular faces,
centered islands and both corner bumps, boundary layer stretching, north/east
diffusion ghosts, three-step agreement with the reference velocity history,
a manufactured basin at advective CFL 0.4, and a finite SI wind run at CFL 0.4.
