# Wind-run energy diagnostics

`plot_energies.py` implements the energy definitions on page 3 of
Shevchenko & Berloff (2015), *Ocean Modelling* 94, 1–14,
DOI: [10.1016/j.ocemod.2015.07.018](https://doi.org/10.1016/j.ocemod.2015.07.018).
The paper has three/six layers; this script specializes its definitions to the
single interface of this repository's two-layer model. It does not reproduce
the paper's model parameters or claim its numerical results.

From the repository root:

```bash
.venv/bin/python tutorials/qg2/plot_energies.py /path/to/wind_SI_N128_run \
  --mean-start-year 50 --mean-end-year 100
```

Omit the mean-window options to use all available saved times (including
spin-up). `--mean-end-year` defaults to the final saved time. The averaging
window must lie within the actual saved data; interrupted runs are supported.
One snapshot is allowed, but cannot provide eddy statistics and gives EPE/EKE=0.

To process multiple folders, pass several paths or a quoted glob:

```bash
.venv/bin/python tutorials/qg2/plot_energies.py 'tutorials/qg2/wind_SI_N*'
```

The script also accepts obstacle-free `wind2_SI_*` runs. Default outputs are in
each run's `energy_diagnostics/` subdirectory:

- `energies.png`: PE, KE, EPE, EKE time series. KE/EKE show both layers and their sum.
- `energies.csv`: the numerical time series (time in years).
- `energy_summary.json`: units, reference window, depth fractions, temporal
  means, energies of the mean flow, and decomposition residuals.

## Time-mean flow maps

Add `--plot-mean-flow` to also create a static four-panel plot in the GIF layout:
top/bottom rows are layers 1/2; left/right columns are the time-mean
streamfunction/PV anomaly. The averaging interval is exactly the one specified
by `--mean-start-year` and `--mean-end-year`, with the same clipped trapezoidal
weights as the energy calculations. PV is averaged directly from saved `q`,
not recomputed by solving a new initial-value problem.

```bash
.venv/bin/python tutorials/qg2/plot_energies.py /path/to/wind_SI_N128_run \
  --mean-start-year 50 --mean-end-year 100 --plot-mean-flow
```

Additional outputs in the same directory:

- `mean_flow.png`: streamfunction and PV anomaly for both layers, with the
  obstacle shaded gray, diverging red/blue colours, and symmetric colour scales
  shared by the two layers for each variable.
- `mean_flow.npz`: mean `psi`, mean `q`, `solid_mask`, node coordinates `x/y`,
  `mean_window_years`, and scalar strings `psi_units/q_units/coordinate_units`.
  The arrays use the original (2,N,N) layout, without padded plotting endpoints.
  Reference length, velocity and time scales are included for unit conversion.

By default coordinates are metres, psi is m^2/s and q is s^-1. With
`--dimensionless`, coordinates are divided by L0, psi by L0*U0, and q is
multiplied by T0. Colour limits are automatically determined from the mean
fields, not from instantaneous extremes in the GIF. Override them with, e.g.,
`--pv-limit 1e-5` for a PV range of [-1e-5,+1e-5] s^-1. `--psi-limit` works
likewise in m^2/s. With `--dimensionless`, both limits are dimensionless.
Values outside the limits saturate; saved mean data are not clipped.

The maps restore omitted north/east psi endpoints as zeros. When the metadata
specify Navier slip, omitted wall PV is reconstructed from mean psi using the
solver's linear wall closure; legacy-mask runs use zero there. Solid nodes are
masked rather than plotted as fluid. This does not alter the saved fields.
The energy-only mode is unchanged and still works without saved PV; the
optional mean-flow plot requires a finite `q` field in each contributing frame.

Use `--output-dir PATH` to choose another output location. With several input
runs this location receives one subdirectory per run. Saved fields and run
metadata are never modified. Only NumPy and Matplotlib are required; importing
the numerical solver or building its inverses is unnecessary.

## Two-layer definitions and units

Let B=Lx*Ly, h1=H1/(H1+H2), h2=H2/(H1+H2). The PV coupling requires
H1*F1=H2*F2, so the saved stretching coefficients determine
h1=F2/(F1+F2), h2=F1/(F1+F2). For the wind defaults these are 0.2 and 0.8.
The wind-conversion parameter `upper_depth` need not be used to guess an
independent depth ratio. Absolute total depth cancels from these diagnostics.

```
P(t)   = h1*F1/(2*B) * integral_fluid (psi1-psi2)^2 dA
K_i(t) = h_i/(2*B)   * integral_fluid |grad psi_i|^2 dA
```

The potential term counts the ONE interface ONCE, since h1*F1=h2*F2.
The paper's multilayer expression has inconsistent summation/index labels;
the two-layer specialization uses this unambiguous shared interface coefficient.
The normalization is B times total depth, as on page 3, not each layer's
individual volume. Consequently K_i are depth-weighted contributions to the
whole-column specific energy, not unweighted layer mean squared speeds.

Physical streamfunction is reconstructed as psi_SI=psi_saved*L0*U0 for
dimensionless snapshots. F1/F2 are read in m^-2. Outputs default to
m^2/s^2 = J/kg; multiplying by density or total depth is NOT needed for these
paper-normalized specific energies. `--dimensionless` divides energies by
the saved reference U0^2, not an assumed energy scale from the paper.

The integrals exclude solid regions, while their denominator B remains the
full box area. Zero streamfunction is enforced on solid/regular-wall nodes;
inconsistent nonzero boundary data are rejected. The omitted north/east wall
nodes are restored as zeros. KE uses squared edge differences with transverse
trapezoidal quadrature, consistent with the solver's five-point Laplacian.
This is preferable to squaring collocated central differences, which can
hide alternating-grid contributions. It is a discrete approximation of the
paper's gradient integral, not an exact continuum gradient.

## Eddy reference and time averaging

A fixed mean streamfunction is computed over the selected physical-time window,
using trapezoidal snapshot weights (including fractional endpoint intervals).
The same weights are used for every reported temporal average. This handles
unequal output intervals and the shorter final output interval without giving
every snapshot equal weight. Only two streamfunction passes are needed, so
memory use does not grow with the number of frames.

EPE/EKE apply the same quadratic forms to psi'=psi-mean(psi). The script plots
these instantaneous quadratic quantities; the paper's overbarred EPE/EKE are
their time averages, recorded in the JSON summary. Likewise `time_mean_total`
corresponds to the paper's overbarred P and K_i, while `mean_flow` is the
energy evaluated on mean(psi). The script verifies

```
time_mean_total = mean_flow + time_mean_eddy
```

This identity holds AFTER averaging, not at each instant (cross terms need
not vanish instantaneously). With a late averaging window, the plotted eddy
energy at t=0 can be nonzero even for an initially resting flow: the initial
state differs from the late mean. The shaded plot region is the reference
window. All saved times remain visible in the plots.

Choose a window after spin-up when estimating equilibrated eddy statistics;
the script does not determine statistical equilibrium automatically. It can
only diagnose saved snapshots: coarse output intervals can alias transient
variability, and changing the averaging window can change eddy energies.
