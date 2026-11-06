"""Rectangular wind basin: no obstacle inverse and exactly-once SI scaling."""
import json

import numpy as np
import pytest

from pequod.solvers.qg_units import QG_SCALES as S, SECONDS_PER_YEAR
from tutorials.qg2.wind_experiment import make_wind_solver as bumped_solver
from tutorials.qg2.wind_experiment2 import make_wind_solver, run, wind_source_amplitude


def test_plain_basin_has_no_capacitance_matrix_and_same_forcing():
    source = wind_source_amplitude(wind_stress=.1)
    solver = make_wind_solver(32, 180000., source, cfl=.4)
    reference = bumped_solver(32, 180000., source, bump_cells=4, cfl=.4)
    solver.initialize()
    engine = solver.dimensionless
    for inverse in (engine._barotropic_solver, engine._baroclinic_solver):
        assert inverse.backend == "dst"
        assert inverse._dst_boundary.size == 0
        assert not hasattr(inverse, "_capacitance_factor")
        assert not hasattr(inverse, "_dst_green")
        assert not hasattr(inverse, "_factor")
    assert engine.active_mask[1:, 1:].all()
    assert not engine.active_mask[0].any()
    assert not engine.active_mask[:, 0].any()
    for layer, other in zip(engine.layers, reference.dimensionless.layers):
        assert not layer.has_obstacle
        assert not layer.obstacle_mask.any()
        assert layer.points.shape == (0, 2)
        assert layer.Cmat is None
        assert layer.grid_boundary_green is None
        assert all(indices.size == 0 for indices in
                   (layer.west_bi, layer.east_bi, layer.north_bi, layer.south_bi))
        assert layer.slip_length == 30000./S.length
        np.testing.assert_array_equal(layer.external_forcing, other.external_forcing)


def test_wind_source_scaled_by_reference_time_squared_exactly_once():
    source = wind_source_amplitude(wind_stress=.1)
    solver = make_wind_solver(32, 180000., source, beta=0., drag=0., nu=0.)
    engine = solver.dimensionless
    physical_source = -source*np.cos(np.pi*engine.layers[0].Y[:-1, :-1])
    np.testing.assert_allclose(engine.layers[0].external_forcing,
                               physical_source*S.time**2, rtol=5.e-15, atol=0.)
    np.testing.assert_allclose(solver.layers[0].external_forcing,
                               physical_source, rtol=5.e-15, atol=0.)
    np.testing.assert_array_equal(engine.layers[1].external_forcing, 0.)
    solver.initialize()
    solver.final_time = 5.  # Seconds: first step starts from rest.
    solver.step()
    mask = engine.active_mask
    # Independently test the physical tendency, not just the stored forcing.
    np.testing.assert_allclose(solver.potential_vorticity[0, mask],
                               5*physical_source[mask], rtol=2.e-14, atol=1.e-28)
    np.testing.assert_array_equal(solver.potential_vorticity[1, mask], 0.)


def test_obstacle_free_run_records_geometry_and_outputs(tmp_path, capsys):
    out = run(n=32, radius=180000., final_time=5/SECONDS_PER_YEAR,
              interval=2., output=tmp_path/"plain", gif=False)
    config = json.loads((out/"config.json").read_text())
    assert config["obstacle"] is False
    assert config["obstacle_size"] is None
    assert config["bump_cells"] is None
    assert config["wind_stress_profile"] == "tau_x=W*sin(pi*y/L), tau_y=0"
    assert config["dimensionless"]["wind"] == pytest.approx(config["wind"]*S.time**2)
    records = [json.loads(s) for s in (out/"diagnostics.jsonl").read_text().splitlines()]
    assert records[-1]["t_seconds"] == pytest.approx(5.)
    assert records[-1]["max_velocity_m_s"] > 0.
    assert "PV inverses: dst, dst" in capsys.readouterr().out
    with np.load(out/"frame_00003.npz") as frame:
        assert frame["q"].shape == frame["psi"].shape == (2, 32, 32)
        assert np.isfinite(frame["q"]).all()
