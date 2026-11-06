"""Independent modal, coupled-stencil, boundary and evolution checks."""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from pequod import grid, qg2
qg2 = partial(qg2, units="dimensionless")
from tutorials.qg2.main import make_case, northwest_bump_wind_source


def test_wind_curl_sign_and_upper_layer_only():
    y = np.array([0., .25, .5, .75, 1.])
    source = northwest_bump_wind_source(np.zeros_like(y), y)
    np.testing.assert_allclose(source[0], [0., -.1, 0., .1, 0.], atol=1.e-15)
    np.testing.assert_array_equal(source[1], 0.)
    eps = 1.e-6
    stress = lambda y: -.1*np.cos(2*np.pi*y)/(2*np.pi)
    np.testing.assert_allclose(source[0], -(stress(y+eps)-stress(y-eps))/(2*eps), atol=1.e-11)


def test_wind_radius_and_amplitude_configuration():
    # Small grid only tests parameter plumbing, not deformation-scale resolution.
    solver, _ = make_case(16, "basin_northwest_bump_wind", deformation_radius=.01,
                          wind_amplitude=1., pv_diffusivity=5.e-6)
    np.testing.assert_allclose(solver.coupling, [8000., 2000.])
    for layer in solver.layers:
        assert layer.nu == 5.e-6
    layer = solver.layers[0]
    expected = northwest_bump_wind_source(layer.X[:-1, :-1], layer.Y[:-1, :-1], 1.)
    np.testing.assert_allclose(layer.external_forcing, expected[0])
    np.testing.assert_array_equal(solver.layers[1].external_forcing, 0.)
    with pytest.raises(ValueError, match="deformation_radius"):
        make_case(16, "basin_northwest_bump_wind", deformation_radius=0.)


def test_small_northwest_wind_bump_geometry():
    solver, _ = make_case(128, "basin_northwest_bump_wind", deformation_radius=.05,
                          wind_amplitude=.1, pv_diffusivity=3.e-4, bump_cells=4)
    layer = solver.layers[0]
    assert layer.obstacle_width == layer.obstacle_height == 4/128
    np.testing.assert_allclose(layer.obstacle_center, [2/128, 1-2/128])
    assert layer.obstacle_mask[-1, 0]
    assert layer.obstacle_mask[124, 4]
    assert not layer.obstacle_mask[123, 4]
    assert not layer.obstacle_mask[124, 5]


def test_wind_basin_spins_up_both_layers():
    solver, exact = make_case(32, "basin_northwest_bump_wind")
    assert exact is None
    solver.initialize()
    layer = solver.layers[0]
    np.testing.assert_allclose(layer.obstacle_center, [.125, .75])
    assert layer.obstacle_width == .25
    assert layer.obstacle_height == .5
    np.testing.assert_array_equal(solver.potential_vorticity, 0.)
    np.testing.assert_array_equal(solver.psi_c, 0.)
    solver.final_time = 1.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solver.solve(visual=-1)
    assert np.isfinite(solver.potential_vorticity).all()
    for layer in solver.layers:
        assert np.max(abs(layer.psi_c)) > 1.e-6
        np.testing.assert_array_equal(layer.psi_c[~solver.active_mask], 0.)
        fx, fy = layer.physical_flux()
        assert np.max(abs(fx[:, layer.obstacle_faces_x])) == 0.
        assert np.max(abs(fy[:, layer.obstacle_faces_y])) == 0.
        assert np.max(abs(fx[..., [0, -1]])) == 0.
        assert np.max(abs(fy[..., [0, -1], :])) == 0.


def zero_velocity(x, y):
    return np.zeros((2,)+x.shape), np.zeros((2,)+y.shape)


def test_channel_nonhomogeneous_coupled_inversion():
    mesh = grid(64, 32, 0., 0., 16., 8.)
    mesh.boundary_conditions = {"Channel": (None, (1., 0., 0.), None, (1., 0., 0.))}
    U = np.array([1., .8])[:, None, None]
    F = np.array([.2, .05])[:, None, None]
    def psi(x, y):
        eta = y/8
        return -8*U*(3*eta**2-2*eta**3)
    def pv(x, y):
        p = psi(x, y)
        return 6*U*(2*y/8-1)/8 + F*(p[::-1]-p)
    solver = qg2(mesh, obstacle=False, F1=.2, F2=.05,
                 channel_throughflow=(1., .8), no_slip=True)
    solver.set_potential_vorticity_ivp(pv, lambda x, y: (6*U*(y/8)*(1-y/8), np.zeros((2,)+x.shape)))
    solver.initialize()
    layer = solver.layers[0]
    np.testing.assert_allclose(solver.psi_c, psi(layer.X[:-1,:-1], layer.Y[:-1,:-1]), atol=1.e-11)


def test_channel_wake_transport_island_and_evolution():
    solver, exact = make_case(32, "channel_square_island")
    assert exact is None
    solver.initialize()
    for layer, U in zip(solver.layers, [1., .8]):
        assert layer._channel_psi_south-layer._channel_psi_north == 8*U
        np.testing.assert_allclose(layer.psi_c[layer.obstacle_mask], -4*U, atol=1.e-12)
    solver.final_time = .5
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solver.solve(visual=-1)
    assert np.isfinite(solver.potential_vorticity).all()
    for layer in solver.layers:
        fx, fy = layer.physical_flux()
        assert np.max(abs(fx[:, layer.obstacle_faces_x])) == 0.
        assert np.max(abs(fy[:, layer.obstacle_faces_y])) == 0.
        # A face away from the corners uses Thom vorticity plus stretching.
        j, k = layer.west_bi[len(layer.west_bi)//2]
        other = solver.layers[1-layer.layer_index]
        expected = (2*(layer.psi_c[j,k-1]-layer._obstacle_streamfunction)/solver.domain.dx**2
                    + solver.coupling[layer.layer_index]*(
                        other._obstacle_streamfunction-layer._obstacle_streamfunction))
        np.testing.assert_allclose(layer.potential_vorticity[j,k], expected, atol=1.e-10)
    p = solver.psi_c
    # Include the omitted north Dirichlet values in an independent stencil.
    lap = laplacian(p, solver.domain.dx, solver.domain.dy)
    for i, layer in enumerate(solver.layers):
        lap[i,-1] += (layer._channel_psi_north-p[i,0])/solver.domain.dy**2
        lap[i] += solver.coupling[i]*(p[1-i]-p[i])
    np.testing.assert_allclose(lap[:,solver.active_mask], solver.potential_vorticity[:,solver.active_mask], atol=2.e-10)


def laplacian(p, dx, dy):
    return ((np.roll(p, 1, -1)-2*p+np.roll(p, -1, -1))/dx**2
            + (np.roll(p, 1, -2)-2*p+np.roll(p, -1, -2))/dy**2)


@pytest.mark.parametrize("F1,F2", [(1., 1.), (.7, 2.3)])
def test_periodic_coupled_inversion_of_independent_discrete_modes(F1, F2):
    solver = qg2(grid(32, 32, 0., 0., 2*np.pi, 2*np.pi),
                 obstacle=False, F1=F1, F2=F2)
    x, y = solver.layers[0].X[:-1, :-1], solver.layers[0].Y[:-1, :-1]
    # Includes opposite layer means (constant baroclinic mode).
    psi = np.array([np.cos(x)*np.cos(2*y)+F1*.2,
                    .3*np.sin(3*x)*np.cos(y)-F2*.2])
    pv = laplacian(psi, solver.domain.dx, solver.domain.dy)
    pv[0] += F1*(psi[1]-psi[0])
    pv[1] += F2*(psi[0]-psi[1])
    solver.set_potential_vorticity_ivp(lambda x, y: pv, zero_velocity)
    solver.initialize()
    np.testing.assert_allclose(solver.psi_c, psi, atol=2.e-13)
    np.testing.assert_allclose(solver.relative_vorticity,
                               laplacian(psi, solver.domain.dx, solver.domain.dy),
                               atol=3.e-13)


def test_periodic_barotropic_mean_is_checked():
    solver = qg2(grid(16, 16), obstacle=False)
    solver.set_potential_vorticity_ivp(
        lambda x, y: np.ones((2,)+x.shape), zero_velocity)
    with pytest.raises(ValueError, match="zero mean"):
        solver.initialize()


@pytest.mark.parametrize("case", ["periodic_square_hole", "channel_square_island_manufactured",
                                  "basin_manufactured", "basin_northwest_manufactured"])
def test_manufactured_two_layer_inversion_and_steady_evolution(case):
    errors = []
    for n in (32, 64):
        solver, exact = make_case(n, case)
        solver.initialize()
        layer = solver.layers[0]
        expected_psi, expected_pv, *_ = exact(layer.X[:-1, :-1], layer.Y[:-1, :-1])
        mask = solver.active_mask
        initial_error = [np.linalg.norm((p-e)[mask])/np.linalg.norm(e[mask])
                         for p, e in zip(solver.psi_c, expected_psi)]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solver.solve(visual=-1)
        errors.append(initial_error)
        assert all(layer.t == solver.t for layer in solver.layers)
        psi = solver.psi_c
        reconstructed_pv = laplacian(psi, solver.domain.dx, solver.domain.dy)
        for i in range(2):
            reconstructed_pv[i] += solver.coupling[i]*(psi[1-i]-psi[i])
        np.testing.assert_allclose(solver.potential_vorticity[:, mask],
                                   reconstructed_pv[:, mask], atol=2.e-11)
        for i, layer in enumerate(solver.layers):
            assert np.linalg.norm((layer.psi_c-expected_psi[i])[mask])/np.linalg.norm(expected_psi[i][mask]) < .04
            assert np.linalg.norm((layer.potential_vorticity-expected_pv[i])[mask])/np.linalg.norm(expected_pv[i][mask]) < .02
            np.testing.assert_array_equal(layer.psi_c[~mask], 0.)
            fx, fy = layer.physical_flux()
            if layer.has_obstacle:
                assert np.max(abs(fx[:, layer.obstacle_faces_x])) == 0
                assert np.max(abs(fy[:, layer.obstacle_faces_y])) == 0
            mode = next(iter(layer.domain.boundary_conditions))
            if mode != "Periodic":
                assert np.max(abs(fy[..., [0, -1], :])) == 0
            if mode == "Basin":
                assert np.max(abs(fx[..., [0, -1]])) == 0
    assert np.all(np.array(errors[0])/np.array(errors[1]) > 3.8)


def test_only_lower_layer_drag_uses_relative_vorticity():
    solver = qg2(grid(16, 16), obstacle=False, F1=1., F2=2.,
                 linear_drag=(0., .2))
    solver.layers[0].psi_c[...] = .3
    solver.layers[1].psi_c[...] = -.1
    solver.layers[0].solutions[0] = .7
    solver.layers[1].solutions[0] = .5
    for layer in solver.layers:
        layer.update_dynamic_forcing()
    np.testing.assert_allclose(solver.layers[0].forcing, 0.)
    np.testing.assert_allclose(solver.layers[1].forcing, -.2*(.5+2*(-.1-.3)))


def test_unforced_basin_bump_and_output(tmp_path):
    solver, _ = make_case(32, "basin_northwest_bump")
    solver.final_time = .2
    solver.data_dir = tmp_path
    solver.initialize()
    initial = solver.potential_vorticity.copy()
    solver.solve(save_freq=3, visual=-1)
    assert np.max(abs(solver.potential_vorticity-initial)) > 1.e-4
    assert np.isfinite(solver.potential_vorticity).all()
    final = np.load(sorted(tmp_path.glob("qg2_*.npz"))[-1])
    assert final["q"].shape == (2, 32, 32)
    assert final["t"] == solver.final_time
    np.testing.assert_allclose(final["psi"], solver.psi_c)


def test_velocity_initialization_barotropic_limit():
    solver = qg2(grid(32, 32, 0., 0., 2*np.pi, 2*np.pi), obstacle=False)
    def velocity(x, y):
        return np.stack([np.sin(y), np.sin(y)]), np.stack([-np.sin(x), -np.sin(x)])
    solver.set_ivp(velocity)
    solver.initialize()
    np.testing.assert_allclose(solver.psi_c[0], solver.psi_c[1], atol=1.e-14)
    np.testing.assert_allclose(solver.potential_vorticity, solver.relative_vorticity, atol=1.e-14)


def test_four_panel_animation(tmp_path):
    from PIL import Image

    solver, _ = make_case(16, "basin_northwest_bump")
    solver.data_dir = tmp_path
    with pytest.raises(ValueError, match="save_freq"):
        solver.save_animation(tmp_path / "missing.gif")
    # An old output must not become part of this run's animation.
    np.savez(tmp_path / "qg2_999999.npz", q=np.nan, psi=np.nan)
    solver.final_time = .05
    solver.solve(save_freq=1, visual=-1)
    with pytest.raises(ValueError, match="extension"):
        solver.save_animation(tmp_path / "bad.png")
    output = solver.save_animation(tmp_path / "test.gif", fps=5, dpi=50)
    with Image.open(output) as gif:
        assert gif.size == (500, 400)
        assert gif.n_frames == len(solver._saved_frames)
        assert gif.n_frames >= 2
        assert gif.info["duration"] == 200
    assert (tmp_path / "qg2_999999.npz").exists()


@pytest.mark.parametrize("baroclinic", [False, True])
def test_coupled_rossby_mode_propagates_with_expected_frequency(baroclinic):
    solver = qg2(grid(64, 64, 0., 0., 2*np.pi, 2*np.pi),
                 obstacle=False, F1=1., F2=2., beta=1.)
    weights = np.array([1/3, -2/3] if baroclinic else [1., 1.])[:, None, None]
    denominator = 5. + (3. if baroclinic else 0.)
    omega = -1./denominator
    amplitude = .01
    def psi(x, y, t=0.):
        return amplitude*weights*np.cos(x+2*y-omega*t)
    def velocity(x, y):
        factor = amplitude*weights*np.sin(x+2*y)
        return 2*factor, -factor
    solver.set_potential_vorticity_ivp(lambda x, y: -denominator*psi(x, y), velocity)
    solver.final_time = 1.
    solver.solve(visual=-1)
    layer = solver.layers[0]
    expected = psi(layer.X[:-1, :-1], layer.Y[:-1, :-1], solver.t)
    assert np.linalg.norm(solver.psi_c-expected)/np.linalg.norm(expected) < .01
