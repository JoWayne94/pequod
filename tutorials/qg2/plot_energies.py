"""Plot two-layer wind-run energetics from saved streamfunctions.

Definitions: Shevchenko & Berloff, Ocean Modelling 94 (2015), p. 3,
doi:10.1016/j.ocemod.2015.07.018, specialized to ONE two-layer interface.
For horizontal box area B=Lx*Ly and h_i=H_i/(H1+H2):

    P(t)   = h1*F1/(2*B) * integral (psi1-psi2)**2 dA
    K_i(t) = h_i/(2*B)   * integral |grad psi_i|**2 dA

Infer h1=F2/(F1+F2), h2=F1/(F1+F2), so h1*F1=h2*F2. Do not
double-count the interface or use the paper's three-/six-layer depths.
The denominator remains the full box area (the paper's A=B*H), including
when a bump removes fluid. Energies are mass-specific: m^2/s^2 = J/kg.
EPE and EKE use psi' = psi - time_mean(psi), with a FIXED averaging window.
Their time means and the mean-flow energies obey total = mean-flow + eddy.
Instantaneous total energy need NOT equal mean-flow energy plus eddy energy
because the instantaneous cross term generally does not vanish.

Snapshots are streamed in two field passes. Unequal time intervals use clipped
trapezoidal snapshot weights, consistently for fields and quadratic energies.
KE uses edge differences, including the omitted zero north/east wall nodes;
this positive quadratic form is consistent with the five-point PV inverse.
Solid/wall psi must be zero. Snapshot PV is needed only with --plot-mean-flow.

Examples (run from the repository root):
  .venv/bin/python tutorials/qg2/plot_energies.py PATH_TO_RUN
  .venv/bin/python tutorials/qg2/plot_energies.py 'tutorials/qg2/wind_SI_N*' \
      --mean-start-year 50 --mean-end-year 100 --plot-mean-flow

Default: all available times define the mean, including spin-up if present.
Writes energies.png, energies.csv and energy_summary.json, without modifying
input files. --dimensionless divides energies by this run's reference U0^2;
that scale is not assumed to match the paper's plotted nondimensional scale.
--plot-mean-flow also writes mean_flow.png (GIF layout: rows are layers,
columns are mean streamfunction and mean PV anomaly) and mean_flow.npz.
Also accepts wind2_SI_* folders and interrupted runs with saved snapshots.
"""
import argparse
import csv
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import warnings

import numpy as np


@dataclass
class EnergyGrid:
    n: int
    length: float
    velocity_scale: float
    time_scale: float
    seconds_per_year: float
    psi_scale: float
    time_to_years: float
    fractions: np.ndarray
    interface_coefficient: float
    solid: np.ndarray
    pv_scale: float = 1.
    slip_length: float | None = None

    @classmethod
    def from_config(cls, config):
        if config.get("input_units", "SI") != "SI":
            raise ValueError("Expected an SI wind-run config.json.")
        n = int(config["n"])
        if n < 2 or n != config["n"]:
            raise ValueError("Configuration n must be an integer >= 2.")
        L = float(config["reference_length_m"])
        U = float(config["reference_velocity_m_s"])
        if not all(np.isfinite(x) and x > 0 for x in (L, U)):
            raise ValueError("Reference length and velocity must be positive and finite.")
        T = float(config.get("reference_time_s", L/U))
        year = float(config.get("seconds_per_year", 365.25*86400.))
        if not all(np.isfinite(x) and x > 0 for x in (L, U, T, year)):
            raise ValueError("Reference scales and seconds_per_year must be positive and finite.")
        if not np.isclose(T, L/U, rtol=1.e-10):
            raise ValueError("Inconsistent reference time: require T0=L0/U0.")
        Fs = []
        for name in ("F1", "F2"):
            value = config.get(name)
            if value is None:
                value = config["dimensionless"][name]/L**2
            Fs.append(float(value))
        F1, F2 = Fs
        if not all(np.isfinite(f) and f > 0 for f in Fs):
            raise ValueError("F1 and F2 must be positive SI stretching coefficients (m^-2).")
        frame_units = config.get("frame_units", "dimensionless")
        if frame_units not in {"dimensionless", "SI"}:
            raise ValueError(f"Unsupported frame_units: {frame_units!r}")
        solid = np.zeros((n, n), dtype=bool)
        # Older format-2 folders have obstacle_size but no obstacle flag.
        obstacle = config.get("obstacle", config.get("obstacle_size") is not None)
        if obstacle not in (False, None, "none"):
            if obstacle not in (True, "northwest_bump"):
                raise ValueError(f"Unsupported wind-run obstacle: {obstacle!r}")
            size = config.get("obstacle_size")
            if size is None:
                cells = config.get("bump_cells")
                size = (.25*L, .5*L) if cells is None else (cells*L/n,)*2
            w, h = map(float, size)
            if not all(np.isfinite(v) and 0 < v < L for v in (w, h)):
                raise ValueError("Invalid obstacle_size.")
            if not np.allclose(np.array([w, h])*n/L, np.rint(np.array([w, h])*n/L), atol=1.e-9):
                raise ValueError("Obstacle must align with grid nodes.")
            x = np.arange(n)*L/n
            solid = (x[None, :] <= w+L*1.e-12) & (x[:, None] >= L-h-L*1.e-12)
        return cls(n, L, U, T, year, L*U if frame_units == "dimensionless" else 1.,
                   T/year if frame_units == "dimensionless" else 1/year,
                   np.array([F2, F1])/(F1+F2), F1*F2/(F1+F2), solid,
                   1/T if frame_units == "dimensionless" else 1., config.get("slip_length_m"))

    def read_psi(self, path):
        with np.load(path, allow_pickle=False) as frame:
            psi = np.array(frame["psi"], dtype=np.float64)*self.psi_scale
        if psi.shape != (2, self.n, self.n) or not np.isfinite(psi).all():
            raise ValueError(f"Invalid streamfunction in {path}: expected finite (2,N,N).")
        wall = self.solid.copy()
        wall[0] = wall[:, 0] = True
        tolerance = 1.e-10*max(1., float(np.max(np.abs(psi))))
        if np.max(np.abs(psi[:, wall])) > tolerance:
            raise ValueError(f"Nonzero wall/solid streamfunction in {path}; this diagnostic expects wind-basin psi=0.")
        psi[:, wall] = 0.
        return psi

    def read_pv(self, path):
        with np.load(path, allow_pickle=False) as frame:
            if "q" not in frame:
                raise ValueError(f"--plot-mean-flow requires saved PV 'q'; missing in {path}.")
            q = np.array(frame["q"], dtype=np.float64)*self.pv_scale
        if q.shape != (2, self.n, self.n) or not np.isfinite(q).all():
            raise ValueError(f"Invalid PV in {path}: expected finite (2,N,N).")
        return q


def frame_time(path, grid):
    with np.load(path, allow_pickle=False) as frame:
        times = []
        for key, scale in (("t_years", 1.), ("t_seconds", 1/grid.seconds_per_year),
                           ("t", grid.time_to_years)):
            if key in frame:
                times.append(float(frame[key])*scale)
    if not times or not np.isfinite(times).all():
        raise ValueError(f"Missing or invalid time in {path}.")
    if not np.allclose(times, times[0], rtol=1.e-10, atol=1.e-10):
        raise ValueError(f"Inconsistent physical/model times in {path}.")
    return times[0]


def discover_frames(folder, grid):
    manifest = folder/"frames.json"
    if manifest.exists():
        names = json.loads(manifest.read_text())
        if not isinstance(names, list) or not all(isinstance(s, str) and Path(s).name == s for s in names):
            raise ValueError("frames.json must list local snapshot filenames.")
        paths = [folder/name for name in names]
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate entries in frames.json.")
    else:
        paths = list(folder.glob("frame_*.npz"))
    if not paths:
        raise ValueError(f"No saved frames in {folder}.")
    samples = sorted((frame_time(p, grid), p) for p in paths)
    times = np.array([t for t, _ in samples])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Snapshot times must be distinct and increasing.")
    return times, [p for _, p in samples]


def averaging_weights(times, start=None, end=None):
    """Normalized trapezoid weights, including clipped endpoint intervals."""
    start = float(times[0] if start is None else start)
    end = float(times[-1] if end is None else end)
    # Accept roundoff at the saved endpoints, not genuinely unavailable data.
    tolerance = 64*np.finfo(float).eps*max(1., abs(times[0]), abs(times[-1]))
    if abs(start-times[0]) <= tolerance:
        start = float(times[0])
    if abs(end-times[-1]) <= tolerance:
        end = float(times[-1])
    if not np.isfinite([start, end]).all() or start < times[0] or end > times[-1] or end < start:
        raise ValueError(f"Mean window [{start:g}, {end:g}] must lie within saved times [{times[0]:g}, {times[-1]:g}].")
    if len(times) == 1:
        warnings.warn("Only one snapshot: eddy energies are zero; no time statistics are available.")
        return np.ones(1), (start, end)
    if end == start:
        raise ValueError("Mean window must have positive duration.")
    weights = np.zeros(len(times))
    for k, (left, right) in enumerate(zip(times[:-1], times[1:])):
        lo, hi = max(left, start), min(right, end)
        if hi <= lo:
            continue
        dt = right-left
        alpha, beta = (lo-left)/dt, (hi-left)/dt
        right_weight = .5*dt*(beta**2-alpha**2)
        weights[k] += hi-lo-right_weight
        weights[k+1] += right_weight
    weights /= weights.sum()
    return weights, (start, end)


def spatial_energies(psi, grid):
    """Return (PE, KE1, KE2) in SI; discrete edge-gradient quadratic form."""
    # Wind basins omit their north/east zero-Dirichlet endpoint nodes.
    p = np.pad(psi, ((0, 0), (0, 1), (0, 1)))
    dx = grid.length/grid.n
    gx = np.diff(p, axis=2)/dx
    gy = np.diff(p, axis=1)/dx
    # Transverse trapezoidal quadrature includes half-weight endpoint rows.
    gx[:, [0, -1]] *= np.sqrt(.5)
    gy[:, :, [0, -1]] *= np.sqrt(.5)
    grad_integral = dx**2*(np.sum(gx*gx, axis=(1, 2))+np.sum(gy*gy, axis=(1, 2)))
    pe = .5*grid.interface_coefficient*dx**2*np.sum((psi[0]-psi[1])**2)/grid.length**2
    ke = .5*grid.fractions*grad_integral/grid.length**2
    return np.array([pe, ke[0], ke[1]])


def analyze_run(folder, *, mean_start_year=None, mean_end_year=None, dimensionless=False,
                plot_mean_flow=False):
    folder = Path(folder).expanduser().resolve()
    config = json.loads((folder/"config.json").read_text())
    grid = EnergyGrid.from_config(config)
    times, paths = discover_frames(folder, grid)
    weights, window = averaging_weights(times, mean_start_year, mean_end_year)
    mean_psi = np.zeros((2, grid.n, grid.n))
    mean_q = np.zeros_like(mean_psi) if plot_mean_flow else None
    for path, weight in zip(paths, weights):
        if weight:
            mean_psi += weight*grid.read_psi(path)
            if plot_mean_flow:
                mean_q += weight*grid.read_pv(path)
    total = np.empty((len(paths), 3))
    eddy = np.empty_like(total)
    for i, path in enumerate(paths):
        psi = grid.read_psi(path)
        total[i] = spatial_energies(psi, grid)
        eddy[i] = spatial_energies(psi-mean_psi, grid)
    mean_flow = spatial_energies(mean_psi, grid)
    scale = grid.velocity_scale**2 if dimensionless else 1.
    total /= scale
    eddy /= scale
    mean_flow /= scale
    total_mean, eddy_mean = weights@total, weights@eddy
    residual = total_mean-mean_flow-eddy_mean
    if not np.allclose(total_mean, mean_flow+eddy_mean, rtol=1.e-10, atol=1.e-300):
        raise ArithmeticError("Energy decomposition failed: mean(total) != energy(mean psi) + mean(eddy).")
    summary = dict(source="Shevchenko & Berloff (2015), Ocean Modelling 94, p. 3",
                   doi="10.1016/j.ocemod.2015.07.018", run=str(folder), snapshots=len(paths),
                   units="dimensionless (reference U0^2)" if dimensionless else "m^2 s^-2 (J kg^-1)",
                   mean_window_years=list(window), saved_window_years=[float(times[0]), float(times[-1])],
                   averaging="clipped trapezoidal snapshot weights; same weights for psi and energy",
                   reference_velocity_m_s=grid.velocity_scale,
                   layer_depth_fractions=grid.fractions.tolist(),
                   interface_coefficient_m_2=grid.interface_coefficient,
                   horizontal_normalization_area_m2=grid.length**2,
                   solid_grid_nodes=int(grid.solid.sum()),
                   energy_components=["PE", "KE1", "KE2"],
                   time_mean_total=total_mean.tolist(), mean_flow=mean_flow.tolist(),
                   time_mean_eddy=eddy_mean.tolist(), decomposition_residual=residual.tolist(),
                   note="Eddies are relative to the selected fixed mean, including for plotted times outside its window.")
    if plot_mean_flow:
        summary["mean_field_units"] = {"psi": "dimensionless" if dimensionless else "m^2 s^-1",
                                       "q": "dimensionless" if dimensionless else "s^-1",
                                       "coordinates": "dimensionless" if dimensionless else "m"}
    return dict(times=times, total=total, eddy=eddy, mean_flow=mean_flow,
                weights=weights, summary=summary, grid=grid,
                mean_streamfunction=mean_psi, mean_pv=mean_q, dimensionless=dimensionless)


def mean_plot_fields(result):
    """Restore omitted wall nodes and mask the NW solid, without inversion.

    North/east PV follows the same linear Navier wall closure as the solver.
    Old legacy-mask runs have zero PV on these omitted regular walls.
    """
    grid = result["grid"]
    if result["mean_pv"] is None:
        raise ValueError("Analyze with plot_mean_flow=True before plotting mean fields.")
    psi = np.pad(result["mean_streamfunction"], ((0, 0), (0, 1), (0, 1)))
    q = np.pad(result["mean_pv"], ((0, 0), (0, 1), (0, 1)))
    h = grid.length/grid.n
    ell = grid.slip_length
    if ell is not None:
        if np.isnan(ell) or ell < 0:
            raise ValueError("Invalid slip_length_m for plotting wall PV.")
        coefficient = 0. if np.isinf(ell) else 2/(h*(h+2*ell))
        q[:, -1, :-1] = coefficient*result["mean_streamfunction"][:, -1, :]
        q[:, :-1, -1] = coefficient*result["mean_streamfunction"][:, :, -1]
    solid = np.pad(grid.solid, ((0, 1), (0, 1)))
    solid[-1, :-1] = grid.solid[-1]  # The NW bump reaches the omitted north wall.
    if result["dimensionless"]:
        psi /= grid.length*grid.velocity_scale
        q *= grid.time_scale
    mask = np.broadcast_to(solid, psi.shape)
    return np.ma.array(psi, mask=mask), np.ma.array(q, mask=mask)


def write_mean_flow(result, output, *, pv_limit=None, psi_limit=None):
    """GIF-style static mean maps plus mean fields in documented plot units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for limit in (pv_limit, psi_limit):
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError("Mean-field colour limits must be finite and positive.")
    grid = result["grid"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    psi, q = mean_plot_fields(result)
    dimensionless = result["dimensionless"]
    length = 1. if dimensionless else grid.length
    dx = length/grid.n
    active = ~grid.solid.copy()
    active[0] = active[:, 0] = False
    fields = (psi, q)
    limits = [override if override is not None else float(np.max(np.abs(field[:, :-1, :-1][:, active]))) or 1.
              for field, override in zip(fields, (psi_limit, pv_limit))]
    start, end = result["summary"]["mean_window_years"]
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
        try:
            for i in range(2):
                for j, (field, bound) in enumerate(zip(fields, limits)):
                    ax = axes[i, j]
                    artist = ax.imshow(field[i], origin="lower", cmap="RdBu_r",
                        vmin=-bound, vmax=bound, interpolation="nearest",
                        extent=(-dx/2, length+dx/2, -dx/2, length+dx/2))
                    title = "time-mean streamfunction" if j == 0 else "time-mean PV anomaly"
                    ax.set(xlim=(0., length), ylim=(0., length),
                           xlabel="x" if dimensionless else "x (m)",
                           ylabel="y" if dimensionless else "y (m)", title=f"Layer {i+1}: {title}")
                    if grid.solid.any():
                        rows, cols = np.where(grid.solid)
                        width = cols.max()*dx
                        south = rows.min()*dx
                        ax.add_patch(Rectangle((0., south), width, length-south,
                            facecolor="0.65", edgecolor="black", linewidth=.8))
                    label = rf"$\overline{{\psi}}_{i+1}$" if j == 0 else rf"$\overline{{q}}_{i+1}$"
                    if not dimensionless:
                        label += " (m²/s)" if j == 0 else " (s⁻¹)"
                    fig.colorbar(artist, ax=ax, label=label,
                                 extend="both" if np.max(np.abs(field[i])) > bound else "neither")
            fig.suptitle(f"Two-layer QG | time mean: {start:g}-{end:g} years")
            fig.savefig(output/"mean_flow.png", dpi=180)
        finally:
            plt.close(fig)
    psi_factor = 1/(grid.length*grid.velocity_scale) if dimensionless else 1.
    q_factor = grid.time_scale if dimensionless else 1.
    np.savez_compressed(output/"mean_flow.npz", psi=result["mean_streamfunction"]*psi_factor,
        q=result["mean_pv"]*q_factor, solid_mask=grid.solid,
        x=np.arange(grid.n)*dx, y=np.arange(grid.n)*dx,
        mean_window_years=np.array([start, end]),
        reference_length_m=grid.length, reference_velocity_m_s=grid.velocity_scale,
        reference_time_s=grid.time_scale,
        psi_units=result["summary"]["mean_field_units"]["psi"],
        q_units=result["summary"]["mean_field_units"]["q"],
        coordinate_units=result["summary"]["mean_field_units"]["coordinates"])
    return output/"mean_flow.png"


def write_outputs(result, output, *, pv_limit=None, psi_limit=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times, total, eddy = result["times"], result["total"], result["eddy"]
    summary = result["summary"]
    with (output/"energies.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_years", "PE", "KE1", "KE2", "KE_total", "EPE", "EKE1", "EKE2", "EKE_total"])
        for t, e, f in zip(times, total, eddy):
            writer.writerow([t, *e, e[1]+e[2], *f, f[1]+f[2]])
    (output/"energy_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, layout="constrained")
    start, end = summary["mean_window_years"]
    unit = "Dimensionless energy" if summary["units"].startswith("dimensionless") else r"Energy (m$^2$ s$^{-2}$ = J kg$^{-1}$)"
    colors = ("#1779a4", "#dc6b21", "#383e49")
    for ax, values, title, potential in zip(axes.flat, (total, total, eddy, eddy),
            ("Potential energy", "Kinetic energy", "Eddy potential energy", "Eddy kinetic energy"),
            (True, False, True, False)):
        if potential:
            ax.plot(times, values[:, 0], color=colors[0], lw=1.2, label="Instantaneous")
            ax.hlines(result["weights"]@values[:, 0], start, end, color=colors[1], ls="--", label="Window mean")
        else:
            for k in range(2):
                ax.plot(times, values[:, k+1], color=colors[k], lw=1., label=f"Layer {k+1}")
            ax.plot(times, values[:, 1:].sum(axis=1), color=colors[2], lw=1.2, label="Both layers")
        if start > times[0] or end < times[-1]:
            ax.axvspan(start, end, color="#1779a4", alpha=.07)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(alpha=.18)
        ax.set_ylim(bottom=0.)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (years)")
    fig.suptitle(f"Two-layer QG basin energetics | mean: {start:g}-{end:g} years", fontsize=14)
    fig.savefig(output/"energies.png", dpi=180)
    plt.close(fig)
    if result.get("mean_pv") is not None:
        write_mean_flow(result, output, pv_limit=pv_limit, psi_limit=psi_limit)
    return output/"energies.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run folder(s) or quoted glob(s) containing config.json and frames")
    parser.add_argument("--mean-start-year", type=float, help="Start of the fixed eddy-reference mean; default first saved time")
    parser.add_argument("--mean-end-year", type=float, help="End of the mean; default last saved time")
    parser.add_argument("--dimensionless", action="store_true", help="Use dimensionless energies and mean fields based on the run's reference scales")
    parser.add_argument("--plot-mean-flow", action="store_true", help="Also plot mean psi and PV in a GIF-style four-panel figure")
    parser.add_argument("--pv-limit", type=float, help="Symmetric mean-PV colour limit: s^-1, or dimensionless with --dimensionless")
    parser.add_argument("--psi-limit", type=float, help="Symmetric mean-psi colour limit: m^2/s, or dimensionless with --dimensionless")
    parser.add_argument("--output-dir", type=Path, help="Output location; multiple runs get separate named subfolders")
    args = parser.parse_args(argv)
    if args.pv_limit is not None or args.psi_limit is not None:
        if not args.plot_mean_flow:
            parser.error("--pv-limit/--psi-limit require --plot-mean-flow")
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (args.pv_limit, args.psi_limit)):
            parser.error("Colour limits must be finite and positive")
    folders = []
    for pattern in args.runs:
        paths = [Path(p) for p in glob.glob(str(Path(pattern).expanduser()))]
        if not paths:
            parser.error(f"No run folder matches {pattern!r}")
        folders.extend(p.resolve() for p in paths)
    folders = list(dict.fromkeys(folders))
    if args.mean_start_year is None:
        warnings.warn("Mean includes all saved spin-up times unless --mean-start-year is specified.")
    for folder in folders:
        try:
            result = analyze_run(folder, mean_start_year=args.mean_start_year,
                                 mean_end_year=args.mean_end_year, dimensionless=args.dimensionless,
                                 plot_mean_flow=args.plot_mean_flow)
            output = args.output_dir or folder/"energy_diagnostics"
            if args.output_dir and len(folders) > 1:
                output = output/folder.name
            plot = write_outputs(result, output, pv_limit=args.pv_limit, psi_limit=args.psi_limit)
        except (OSError, ValueError, KeyError, ArithmeticError) as error:
            parser.error(f"{folder}: {error}")
        print(f"{folder.name}: {result['summary']['snapshots']} snapshots; plot: {plot.resolve()}", flush=True)
        if args.plot_mean_flow:
            print(f"Mean flow: {(Path(output)/'mean_flow.png').resolve()}", flush=True)


if __name__ == "__main__":
    main()
