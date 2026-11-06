"""Independent checks of dimensional boundaries, callbacks and time evolution."""
import json

import numpy as np
import pytest

from pequod import grid, qg, qg2
from pequod.solvers.qg_units import QG_SCALES as S, SECONDS_PER_YEAR
from tutorials.qg2.wind_experiment import run, wind_source_amplitude


@pytest.mark.parametrize("factory", [qg, qg2])
def test_si_defaults_and_parameter_setters(factory):
    mesh = grid(16, 16, 0., 0., S.length, S.length)
    solver = factory(mesh, obstacle=False)
    assert solver.reference_time == 384_000_000.
    assert solver.domain is mesh
    assert solver.domain.east == S.length
    assert solver.dimensionless.domain.east == 1.
    assert solver.beta == pytest.approx(2.e-11)
    assert solver.dimensionless.beta == pytest.approx(29491.2)
    np.testing.assert_allclose(solver.dimensionless.nu, .0026041666666666665)
    np.testing.assert_allclose(solver.nu, 100.)
    np.testing.assert_allclose(solver.linear_drag, (0., 4.e-8) if factory is qg2 else 4.e-8)
    solver.beta = 1.e-11
    solver.nu = 200.
    solver.linear_drag = 8.e-8
    solver.final_time = SECONDS_PER_YEAR
    assert solver.dimensionless.beta == pytest.approx(14745.6)
    np.testing.assert_allclose(solver.dimensionless.nu, 200/(S.length*S.velocity))
    np.testing.assert_allclose(solver.dimensionless.linear_drag, 30.72)
    assert solver.dimensionless.final_time == pytest.approx(SECONDS_PER_YEAR/S.time)


def test_single_layer_si_discrete_fourier_inversion_and_source():
    n = 32
    radius = S.length/5
    solver = qg(grid(n, n, 0., 0., S.length, S.length), obstacle=False,
                deformation_radius=radius, beta=0., nu=0., linear_drag=0.)
    lap_eigenvalue = -8*np.sin(np.pi/n)**2/(S.length/n)**2
    def psi(x, y):
        return S.streamfunction*np.cos(2*np.pi*x/S.length)*np.cos(2*np.pi*y/S.length)
    def velocity(x, y):
        k = 2*np.pi/S.length
        return (k*S.streamfunction*np.cos(k*x)*np.sin(k*y),
                -k*S.streamfunction*np.sin(k*x)*np.cos(k*y))
    solver.set_potential_vorticity_ivp(lambda x, y: (lap_eigenvalue-radius**-2)*psi(x, y), velocity)
    solver.set_forcing_terms(f=lambda x, y: np.full_like(x, 2.e-15))
    solver.initialize()
    np.testing.assert_allclose(solver.psi_c, psi(solver.X[:-1,:-1], solver.Y[:-1,:-1]), atol=2.e-9)
    np.testing.assert_allclose(solver.dimensionless.external_forcing, 2.e-15*S.time**2)
    assert solver.deformation_radius == radius
    assert solver.inverse_deformation_radius_squared == pytest.approx(1/radius**2)
    assert solver.max_velocity_m_s == pytest.approx(2*np.pi*S.velocity*np.sin(2*np.pi/n)/(2*np.pi/n))


def test_single_layer_si_near_rest_timestep_has_beta_drag_limit():
    solver = qg(grid(16,16,0.,0.,S.length,S.length), obstacle=False)
    solver.set_ivp(ic=lambda x,y: (np.zeros_like(x), np.zeros_like(y)))
    solver.final_time = SECONDS_PER_YEAR
    solver.initialize()
    assert 0 < solver.dt <= solver.cfl/(abs(solver.beta)*S.length+solver.linear_drag)*(1+1.e-14)


def test_two_layer_si_and_dimensionless_evolution_agree():
    n = 16
    F = np.array([.8, .2])[:,None,None]
    weights = np.array([1., .5])[:,None,None]
    lap = -8*np.sin(np.pi/n)**2*n*n
    def psi(x, y):
        return .01*weights*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    def pv(x, y):
        p = psi(x, y)
        return lap*p+F*(p[::-1]-p)
    def velocity(x, y):
        return (.01*weights*2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y),
                -.01*weights*2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y))
    nd = qg2(grid(n,n), obstacle=False, F1=.8, F2=.2, beta=.1, nu=.001,
              linear_drag=(0., .02), units="dimensionless")
    si = qg2(grid(n,n,0.,0.,S.length,S.length), obstacle=False,
              F1=.8/S.length**2, F2=.2/S.length**2, beta=.1*S.beta,
              nu=.001*S.diffusivity, linear_drag=(0., .02/S.time))
    nd.set_potential_vorticity_ivp(pv, velocity)
    si.set_potential_vorticity_ivp(lambda x,y: pv(x/S.length,y/S.length)/S.time,
        lambda x,y: tuple(a*S.velocity for a in velocity(x/S.length,y/S.length)))
    forcing = lambda x,y: .001*weights*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    nd.set_forcing_terms(forcing)
    si.set_forcing_terms(lambda x,y: forcing(x/S.length,y/S.length)/S.time**2)
    nd.final_time = .01
    si.final_time = .01*S.time
    nd.solve(visual=-1)
    si.solve(visual=-1)
    assert si.t == pytest.approx(.01*S.time)
    np.testing.assert_allclose(si.potential_vorticity*S.time, nd.potential_vorticity, atol=1.e-12)
    np.testing.assert_allclose(si.psi_c/S.streamfunction, nd.psi_c, atol=1.e-12)
    np.testing.assert_allclose(si.max_velocity_by_layer_m_s,
                               nd.max_velocity_by_layer_m_s, atol=1.e-12)
    np.testing.assert_allclose(si.layers[0].psi_c, si.psi_c[0])


def test_channel_si_transport_and_obstacle_boundary_values():
    mesh = grid(64, 32, 0., 0., 16*S.length, 8*S.length)
    mesh.boundary_conditions = {"Channel": (None, (1., 0., 0.), None, (1., 0., 0.))}
    solver = qg2(mesh, obstacle="centered", obstacle_size=(S.length, S.length),
                 F1=.2/S.length**2, F2=.05/S.length**2, beta=0.,
                 nu=100., channel_throughflow=(.01,.008), no_slip=True)
    zeros = lambda x,y: np.zeros((2,)+x.shape)
    solver.set_potential_vorticity_ivp(zeros, lambda x,y: (zeros(x,y), zeros(x,y)))
    solver.initialize()
    for i, (layer, U) in enumerate(zip(solver.layers, [.01,.008])):
        assert layer.channel_throughflow == pytest.approx(U)
        assert layer._channel_psi_north == pytest.approx(-8*S.length*U)
        np.testing.assert_allclose(layer.psi_c[layer.obstacle_mask], -4*S.length*U)
    np.testing.assert_allclose(solver.dimensionless.layers[0].channel_throughflow, 1.)


def test_wind_stress_curl_and_output_units(tmp_path, capsys):
    stress, rho, H = .1, 1025., 500.
    W = wind_source_amplitude(wind_stress=stress, density=rho, upper_depth=H)
    assert W == pytest.approx(np.pi*stress/(rho*H*S.length), rel=1.e-14, abs=0.)
    y = S.length*np.array([.1,.25,.7,.9])
    acceleration = lambda y: stress*np.sin(np.pi*y/S.length)/(rho*H)
    eps = 1.
    np.testing.assert_allclose(-(acceleration(y+eps)-acceleration(y-eps))/(2*eps),
                               -W*np.cos(np.pi*y/S.length), rtol=1.e-8, atol=1.e-24)
    with pytest.raises(ValueError, match="either"):
        wind_source_amplitude(wind=1.e-13, wind_stress=.1)
    output = run(n=32, radius=S.length*6/32, final_time=5/SECONDS_PER_YEAR,
                 interval=2., bump_cells=4, output=tmp_path/"experiment", gif=False)
    config = json.loads((output/"config.json").read_text())
    assert config["format_version"] == 3
    assert config["wind_stress_profile"] == "tau_x=W*sin(pi*y/L), tau_y=0"
    assert config["pv_source_profile"] == "S1=-amplitude*cos(pi*y/L), S2=0"
    assert config["input_units"] == "SI"
    assert config["frame_units"] == "dimensionless"
    assert config["final_time_seconds"] == pytest.approx(5.)
    records = [json.loads(line) for line in (output/"diagnostics.jsonl").read_text().splitlines()]
    assert records[-1]["t_seconds"] == pytest.approx(5.)
    assert records[-1]["max_velocity_m_s"] > 0.
    assert len(records[-1]["layer_max_velocity_m_s"]) == 2
    with np.load(output/"frame_00003.npz") as frame:
        assert frame["t"] == pytest.approx(5/S.time)
        assert frame["t_seconds"] == pytest.approx(5.)
    assert "max_velocity_m_s" in capsys.readouterr().out
