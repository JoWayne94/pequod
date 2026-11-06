"""Paper-normalized two-layer energies, quadrature, units and eddy split."""
import json

import numpy as np
import pytest

from tutorials.qg2.plot_energies import (
    EnergyGrid, analyze_run, averaging_weights, main, spatial_energies, write_outputs,
    mean_plot_fields, write_mean_flow,
)


def config(n=32, frame_units="dimensionless"):
    return dict(n=n, input_units="SI", frame_units=frame_units,
                reference_length_m=2., reference_velocity_m_s=3., reference_time_s=2/3,
                seconds_per_year=10., F1=2., F2=.5, obstacle=False, obstacle_size=None)


def mode(n):
    x = np.arange(n)*np.pi/n
    return np.sin(x[:, None])*np.sin(x[None, :])


def write_run(path, *, units="dimensionless", times=(0., 1., 3.), amplitudes=(1., 2., 4.), save_q=False):
    path.mkdir()
    cfg = config(frame_units=units)
    (path/"config.json").write_text(json.dumps(cfg))
    names = []
    for k, (t, a) in enumerate(zip(times, amplitudes)):
        psi = a*np.array([mode(cfg["n"]), -2*mode(cfg["n"])])
        if units == "dimensionless":
            psi /= 6.
        name = f"frame_{k:05d}.npz"
        values = {}
        if save_q:
            # Distinct amplitudes prove we average saved PV, not reinvert mean psi.
            q = (k+1)**2*1.e-3*np.array([mode(cfg["n"]), -3*mode(cfg["n"])])
            values["q"] = q*(2/3) if units == "dimensionless" else q
        np.savez_compressed(path/name, psi=psi, **values, t=t*10/(2/3) if units == "dimensionless" else t*10,
                            t_years=t, t_seconds=t*10)
        names.append(name)
    (path/"frames.json").write_text(json.dumps(names[::-1]))
    return path


def test_interface_count_depth_weight_and_known_discrete_mode():
    g = EnergyGrid.from_config(config())
    np.testing.assert_allclose(g.fractions, [.2, .8])
    assert g.interface_coefficient == pytest.approx(.4)
    psi = np.array([3*mode(g.n), -2*mode(g.n)])
    pe, k1, k2 = spatial_energies(psi, g)
    eigenvalue = 8*np.sin(np.pi/(2*g.n))**2/(g.length/g.n)**2
    assert pe == pytest.approx(.4*(3+2)**2/8, rel=1.e-14)
    np.testing.assert_allclose([k1, k2], g.fractions*eigenvalue*np.array([9, 4])/8, rtol=1.e-14)


def test_masked_energy_matches_pv_quadratic_form_and_full_box_area():
    cfg = config()
    cfg.pop("obstacle")  # Legacy format-2 metadata infers the NW obstacle.
    cfg["obstacle_size"] = [.5, 1.]
    g = EnergyGrid.from_config(cfg)
    assert np.count_nonzero(g.solid) == 9*16
    rng = np.random.default_rng(17)
    psi = rng.normal(size=(2, g.n, g.n))
    psi[:, 0] = psi[:, :, 0] = 0.
    psi[:, g.solid] = 0.
    p = np.pad(psi, ((0, 0), (1, 1), (1, 1)))
    dx = g.length/g.n
    lap = (p[:, 2:, 1:-1]+p[:, :-2, 1:-1]+p[:, 1:-1, 2:]+p[:, 1:-1, :-2]-4*psi)/dx**2
    q = lap+np.array([2., .5])[:, None, None]*(psi[::-1]-psi)
    pv_energy = -.5*dx**2*np.sum(g.fractions[:, None, None]*psi*q)/g.length**2
    assert spatial_energies(psi, g).sum() == pytest.approx(pv_energy, rel=1.e-14)


def test_unequal_times_and_clipped_mean_window():
    times = np.array([0., 1., 3.])
    w, window = averaging_weights(times)
    np.testing.assert_allclose(w, [1/6, .5, 1/3])
    w, window = averaging_weights(times, .5, 2.)
    np.testing.assert_allclose(w, [.125/1.5, 1.125/1.5, .25/1.5])
    assert window == (.5, 2.)
    assert w@times == pytest.approx(1.25)  # Linear functions integrate exactly.


def test_units_and_eddy_mean_decomposition(tmp_path):
    path = write_run(tmp_path/"wind_SI_N32")
    result = analyze_run(path)
    w = result["weights"]
    amplitudes = np.array([1., 2., 4.])
    energies = spatial_energies(np.array([mode(32), -2*mode(32)]), EnergyGrid.from_config(config()))
    np.testing.assert_allclose(result["total"], amplitudes[:, None]**2*energies)
    np.testing.assert_allclose(result["eddy"], (amplitudes-w@amplitudes)[:, None]**2*energies)
    np.testing.assert_allclose(result["mean_flow"], (w@amplitudes)**2*energies)
    np.testing.assert_allclose(w@result["total"], result["mean_flow"]+w@result["eddy"], rtol=1.e-14)
    si_path = write_run(tmp_path/"physical_frames", units="SI")
    physical = analyze_run(si_path)
    np.testing.assert_allclose(physical["total"], result["total"])
    nondim = analyze_run(path, dimensionless=True)
    np.testing.assert_allclose(nondim["total"], result["total"]/9.)
    # A partial averaging window uses the same weights for all three terms.
    clipped = analyze_run(path, mean_start_year=.5, mean_end_year=2.)
    np.testing.assert_allclose(clipped["weights"]@clipped["total"],
                               clipped["mean_flow"]+clipped["weights"]@clipped["eddy"], rtol=1.e-14)


def test_stationary_flow_has_zero_eddy_energy_and_writes_outputs(tmp_path):
    path = write_run(tmp_path/"steady", amplitudes=(2., 2., 2.))
    result = analyze_run(path)
    np.testing.assert_allclose(result["eddy"], 0., atol=1.e-28)
    plot = write_outputs(result, tmp_path/"plots")
    assert plot.stat().st_size > 10000
    data = np.loadtxt(plot.parent/"energies.csv", delimiter=",", skiprows=1)
    assert data.shape == (3, 9)
    np.testing.assert_allclose(data[:, 4], data[:, 2]+data[:, 3])
    assert json.loads((plot.parent/"energy_summary.json").read_text())["units"] == "m^2 s^-2 (J kg^-1)"


def test_interrupted_run_without_manifest_and_single_frame(tmp_path):
    path = write_run(tmp_path/"partial", times=(0.,), amplitudes=(1.,))
    (path/"frames.json").unlink()
    with pytest.warns(UserWarning, match="one snapshot"):
        result = analyze_run(path)
    np.testing.assert_array_equal(result["eddy"], 0.)


def test_rejects_invalid_times_units_and_solid_values(tmp_path):
    with pytest.raises(ValueError, match="positive duration"):
        averaging_weights(np.array([0., 1.]), 0., 0.)
    with pytest.raises(ValueError, match="within saved"):
        averaging_weights(np.array([0., 1.]), 50.)
    cfg = config()
    cfg["frame_units"] = "unknown"
    with pytest.raises(ValueError, match="frame_units"):
        EnergyGrid.from_config(cfg)
    path = write_run(tmp_path/"bad_wall")
    with np.load(path/"frame_00000.npz") as z:
        values = {key: z[key] for key in z.files}
    values["psi"][0, 0, 1] = 1.
    np.savez_compressed(path/"frame_00000.npz", **values)
    with pytest.raises(ValueError, match="Nonzero wall"):
        analyze_run(path)


def test_cli_multiple_run_glob(tmp_path):
    for name in ("wind_SI_N16_a", "wind_SI_N16_b"):
        write_run(tmp_path/name)
    main([str(tmp_path/"wind_SI_N*"), "--mean-start-year", "1", "--output-dir", str(tmp_path/"plots")])
    assert len(list((tmp_path/"plots").glob("*/energies.png"))) == 2


@pytest.mark.parametrize("units", ["SI", "dimensionless"])
@pytest.mark.parametrize("dimensionless", [False, True])
def test_mean_fields_use_same_clipped_window_and_correct_units(tmp_path, units, dimensionless):
    path = write_run(tmp_path/"run", units=units, save_q=True)
    result = analyze_run(path, mean_start_year=.5, mean_end_year=2.,
                         dimensionless=dimensionless, plot_mean_flow=True)
    weights, _ = averaging_weights(np.array([0., 1., 3.]), .5, 2.)
    expected_psi = (weights@np.array([1., 2., 4.]))*np.array([mode(32), -2*mode(32)])
    expected_q = (weights@np.array([1., 4., 9.]))*1.e-3*np.array([mode(32), -3*mode(32)])
    np.testing.assert_allclose(result["mean_streamfunction"], expected_psi, atol=1.e-15)
    np.testing.assert_allclose(result["mean_pv"], expected_q, atol=1.e-17)
    psi, q = mean_plot_fields(result)
    np.testing.assert_allclose(psi[:, :-1, :-1], expected_psi/(6. if dimensionless else 1.))
    np.testing.assert_allclose(q[:, :-1, :-1], expected_q*((2/3) if dimensionless else 1.))
    # Legacy-mask north/east walls are zero.
    np.testing.assert_array_equal(q[:, -1], 0.)
    np.testing.assert_array_equal(q[:, :, -1], 0.)
    write_outputs(result, tmp_path/"output", pv_limit=.02, psi_limit=5.)
    assert (tmp_path/"output"/"mean_flow.png").stat().st_size > 10000
    with np.load(tmp_path/"output"/"mean_flow.npz", allow_pickle=False) as z:
        np.testing.assert_allclose(z["psi"], expected_psi/(6. if dimensionless else 1.))
        np.testing.assert_allclose(z["q"], expected_q*((2/3) if dimensionless else 1.))
        np.testing.assert_array_equal(z["mean_window_years"], [.5, 2.])
        assert z["q_units"].item() == ("dimensionless" if dimensionless else "s^-1")
        assert z["x"][-1] == pytest.approx(31/32*(1. if dimensionless else 2.))


def test_mean_plot_navier_walls_obstacle_and_gif_layout(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    path = write_run(tmp_path/"run", save_q=True)
    cfg = json.loads((path/"config.json").read_text())
    cfg.update(obstacle="northwest_bump", obstacle_size=[.5, 1.], slip_length_m=.1)
    (path/"config.json").write_text(json.dumps(cfg))
    g = EnergyGrid.from_config(cfg)
    for name in json.loads((path/"frames.json").read_text()):
        with np.load(path/name) as z:
            values = {key: z[key] for key in z.files}
        values["psi"][:, g.solid] = 0.
        np.savez_compressed(path/name, **values)
    result = analyze_run(path, plot_mean_flow=True)
    psi, q = mean_plot_fields(result)
    h = g.length/g.n
    coefficient = 2/(h*(h+2*.1))
    np.testing.assert_allclose(q.data[:, -1, :-1], coefficient*result["mean_streamfunction"][:, -1])
    np.testing.assert_allclose(q.data[:, :-1, -1], coefficient*result["mean_streamfunction"][:, :, -1])
    np.testing.assert_array_equal(q.mask[:, :-1, :-1], np.broadcast_to(g.solid, (2, 32, 32)))
    assert q.mask[:, -1, :9].all()  # NW obstacle reaches the missing north endpoint.
    figures = []
    close = plt.close
    def capture(fig=None):
        if hasattr(fig, "axes"):
            figures.append(fig)
        close(fig)
    monkeypatch.setattr(plt, "close", capture)
    write_mean_flow(result, tmp_path/"out", pv_limit=.01, psi_limit=4.)
    axes = figures[-1].axes[:4]
    for i, ax in enumerate(axes):
        assert f"Layer {i//2+1}" in ax.get_title()
        assert ("streamfunction" if i % 2 == 0 else "PV anomaly") in ax.get_title()
        assert ax.images[0].get_clim() == ((-4., 4.) if i % 2 == 0 else (-.01, .01))
        assert len(ax.patches) == 1


def test_mean_flow_cli_and_missing_q_error(tmp_path):
    path = write_run(tmp_path/"run", save_q=True)
    main([str(path), "--mean-start-year", "1", "--mean-end-year", "3", "--plot-mean-flow",
          "--pv-limit", "0.01"])
    assert (path/"energy_diagnostics"/"mean_flow.png").exists()
    without_pv = write_run(tmp_path/"no_q")
    analyze_run(without_pv)  # Original energy-only use remains supported.
    with pytest.raises(ValueError, match="requires saved PV"):
        analyze_run(without_pv, plot_mean_flow=True)
