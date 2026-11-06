"""Independent bump axes, backward compatibility and recorded geometry."""
import json
import sys

import numpy as np
import pytest

from pequod.solvers.qg_units import QG_SCALES as S, SECONDS_PER_YEAR
from tutorials.qg2 import wind_experiment as wind
from tutorials.qg2.plot_energies import EnergyGrid


@pytest.mark.parametrize("width,height", [(4, 8), (8, 4), (1, 1), (31, 4), (4, 31)])
def test_rectangular_bump_geometry_attaches_to_northwest(width, height):
    n = 32
    solver = wind.make_wind_solver(n, 180000., wind.wind_source_amplitude(),
                                   bump_width_cells=width, bump_height_cells=height)
    solver.initialize()
    expected = np.zeros((n, n), dtype=bool)
    expected[n-height:, :width+1] = True
    for layer in solver.dimensionless.layers:
        np.testing.assert_array_equal(layer.obstacle_mask, expected)
        assert layer.obstacle_width == pytest.approx(width/n)
        assert layer.obstacle_height == pytest.approx(height/n)
        assert layer.obstacle_mask[-1, 0]
        assert np.max(abs(layer.psi_c[layer.obstacle_mask])) == 0.


def test_old_square_and_default_rectangle_are_unchanged():
    assert wind.wind_obstacle_size(128, 4) == (120000., 120000.)
    assert wind.wind_obstacle_size(128) == (.25*S.length, .5*S.length)
    assert wind.wind_obstacle_size(128, bump_width_cells=4, bump_height_cells=8) == (120000., 240000.)
    assert wind.wind_obstacle_size(128, obstacle=False) is None


@pytest.mark.parametrize("kwargs", [
    dict(bump_cells=4, bump_width_cells=4, bump_height_cells=8),
    dict(bump_width_cells=4), dict(bump_height_cells=8),
    dict(bump_width_cells=0, bump_height_cells=8),
    dict(bump_width_cells=4, bump_height_cells=32),
    dict(bump_width_cells=4.5, bump_height_cells=8),
    dict(bump_width_cells=True, bump_height_cells=8),
    dict(bump_cells=-1),
    dict(obstacle=False, bump_width_cells=4, bump_height_cells=8),
])
def test_invalid_or_ambiguous_dimensions_rejected(kwargs):
    with pytest.raises(ValueError):
        wind.make_wind_solver(32, 180000., wind.wind_source_amplitude(), **kwargs)


def test_cli_rectangular_dimensions_metadata_and_postprocessing(tmp_path, monkeypatch, capsys):
    out = tmp_path/"rectangle"
    monkeypatch.setattr(sys, "argv", ["wind_experiment.py", "--resolution", "32", "--radius", "180000",
        "--bump-width-cells", "4", "--bump-height-cells", "8", "--cfl", "0.4",
        "--final-time", str(5/SECONDS_PER_YEAR), "--interval", "2", "--no-gif", "--output", str(out)])
    wind.main()
    cfg = json.loads((out/"config.json").read_text())
    assert cfg["bump_cells"] is None
    assert cfg["bump_width_cells"] == 4
    assert cfg["bump_height_cells"] == 8
    assert cfg["obstacle_size"] == [4*S.length/32, 8*S.length/32]
    mask = EnergyGrid.from_config(cfg).solid
    expected = np.zeros((32, 32), dtype=bool)
    expected[24:, :5] = True
    np.testing.assert_array_equal(mask, expected)
    with np.load(out/"frame_00003.npz") as frame:
        assert np.isfinite(frame["q"]).all()
        assert frame["t_seconds"] == pytest.approx(5.)
        np.testing.assert_array_equal(frame["psi"][:, mask], 0.)


def test_cli_requires_both_axes_without_starting_run(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wind_experiment.py", "--bump-width-cells", "4"])
    with pytest.raises(SystemExit) as error:
        wind.main()
    assert error.value.code == 2
    assert "Specify both" in capsys.readouterr().err
