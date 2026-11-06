"""Independent equivalence and time-accuracy checks for fast QG paths."""
import numpy as np
import pytest

from pequod import grid, qg2
from pequod.gulf import Gulf
from pequod.solvers.incns import IncNavierStokes
from pequod.solvers.qg import _MaskedHelmholtzSolver
from pequod.solvers.qg_units import QG_SCALES
from tutorials.qg2.wind_experiment import make_wind_solver, wind_source_amplitude


@pytest.mark.parametrize("shape", [(24, 32), (32, 32)])
@pytest.mark.parametrize("geometry", ["empty", "northwest", "southeast", "island", "large"])
@pytest.mark.parametrize("coefficient", [0., 455.])
def test_basin_fast_inverse_matches_sparse(shape, geometry, coefficient):
    ny, nx = shape
    mask = np.ones(shape, dtype=bool)
    mask[0] = mask[:, 0] = False
    if geometry == "northwest":
        mask[-4:, :5] = False
    elif geometry == "southeast":
        mask[:5, -4:] = False
    elif geometry == "island":
        mask[ny//2-2:ny//2+3, nx//2-2:nx//2+3] = False
    elif geometry == "large":
        mask[1:-1, 2:-2] = False  # Deliberately exceed the fast-path cutoff.
    options = dict(periodic_x=False, periodic_y=False)
    fast = _MaskedHelmholtzSolver(mask, 2/nx, 3/ny, coefficient, **options)
    sparse = _MaskedHelmholtzSolver(mask, 2/nx, 3/ny, coefficient, backend="sparse", **options)
    assert fast.backend == {"empty": "dst", "large": "sparse"}.get(geometry, "dst-capacitance")
    rng = np.random.default_rng(90)
    for _ in range(2):
        rhs = rng.normal(size=shape)
        expected = sparse.solve(rhs)
        out = np.full(shape, np.nan)
        assert fast.solve(rhs, out=out) is out
        np.testing.assert_allclose(out, expected, rtol=2.e-12, atol=2.e-14)
        np.testing.assert_array_equal(out[~mask], 0.)
        # A residual check, independent of both inversion implementations.
        p = np.pad(out, ((0, 1), (0, 1)))
        lap = ((p[1:ny, 2:nx+1]-2*p[1:ny, 1:nx]+p[1:ny, :nx-1])/(2/nx)**2
               +(p[2:ny+1, 1:nx]-2*p[1:ny, 1:nx]+p[:ny-1, 1:nx])/(3/ny)**2)
        residual = lap-coefficient*out[1:, 1:]-rhs[1:, 1:]
        np.testing.assert_allclose(residual[mask[1:, 1:]], 0., atol=3.e-13)


@pytest.mark.parametrize("mode", ["Periodic", "Channel", "Basin"])
def test_transport_buffers_match_original_cabaret(mode):
    owners = []
    for _ in range(2):
        mesh = grid(16, 16)
        if mode != "Periodic":
            mesh.boundary_conditions = {mode: (None, (1., 0., 0.), None, (1., 0., 0.))
                if mode == "Channel" else ((1., 0., 0.),)*4}
        s = qg2(mesh, obstacle=False, beta=.3, nu=.01, linear_drag=.02, units="dimensionless")
        s.set_ivp(lambda x, y: (np.zeros((2,)+x.shape), np.zeros((2,)+y.shape)))
        s.initialize()
        owners.append(s.dimensionless)
    actual, reference = [s.layers[0] for s in owners]
    rng = np.random.default_rng(51)
    actual.laplacian_u = np.zeros((16, 16))
    reference.laplacian_u = np.zeros((16, 16))
    for name in ("_cons", "_adv_x", "_adv_y", "psi_c", "psi_v"):
        data = rng.normal(size=getattr(actual, name).shape)
        getattr(actual, name)[...] = getattr(reference, name)[...] = data
    actual.dt = reference.dt = .001
    for _ in range(3):
        actual.predictor()
        Gulf.predictor_corrector(reference, lambda: Gulf._predictor(reference))()
        for name in ("_cons", "_Q", "_M_x", "_m_x", "_M_y", "_m_y"):
            np.testing.assert_allclose(getattr(actual, name), getattr(reference, name), atol=1.e-13)
        actual.second_order_extrapolation()
        Gulf.second_order_extrapolation(reference)
        actual.numerical_flux_x(); actual.numerical_flux_y()
        Gulf.numerical_flux_x(reference); Gulf.numerical_flux_y(reference)
        for name in ("_adv_x", "_adv_y"):
            np.testing.assert_allclose(getattr(actual, name), getattr(reference, name), atol=1.e-13)
        actual._average_psi_to_vertices()
        IncNavierStokes._average_psi_to_vertices(reference)
        np.testing.assert_allclose(actual.psi_v, reference.psi_v, atol=1.e-15)
        actual.corrector()
        Gulf.predictor_corrector(reference, reference._corrector)()
        np.testing.assert_allclose(actual.solutions, reference.solutions, atol=1.e-13)


def test_si_wind_source_values():
    amplitude = wind_source_amplitude(wind_stress=.1)
    s = make_wind_solver(32, 180000., amplitude, bump_cells=4)
    e = s.dimensionless
    y = e.layers[0].Y[:-1, :-1]
    np.testing.assert_allclose(e.layers[0].external_forcing,
                               -amplitude*QG_SCALES.time**2*np.cos(np.pi*y))
    np.testing.assert_array_equal(e.layers[1].external_forcing, 0.)


def test_compiled_path_keeps_conservative_source_time_cap():
    s = make_wind_solver(32, 180000., wind_source_amplitude(), bump_cells=4, cfl=.4)
    s.initialize()
    s.final_time = 86400.
    s.step()
    assert s.dt == pytest.approx(.15/(2.e-11*QG_SCALES.length+4.e-8))


@pytest.mark.parametrize("mode", ["Channel", "Basin"])
def test_beta_bound_controls_discrete_gradient_inverse(mode):
    mesh = grid(16, 24, 0., 0., 2., 3.)
    mesh.boundary_conditions = {mode: (None, (1., 0., 0.), None, (1., 0., 0.))
        if mode == "Channel" else ((1., 0., 0.),)*4}
    s = qg2(mesh, obstacle="centered", obstacle_size=(.5, .5),
             beta=100., nu=0., linear_drag=0., units="dimensionless")
    s.set_ivp(lambda x, y: (np.zeros((2,)+x.shape), np.zeros((2,)+y.shape)))
    s.initialize()
    e = s.dimensionless
    l = e.layers[0]
    rng = np.random.default_rng(71)
    for _ in range(4):
        q = rng.normal(size=e.active_mask.shape)*e.active_mask
        l.psi_c[...] = e._barotropic_solver.solve(q)
        l.psi_v_bcs()
        l.alpha_t = 0.
        l.compute_velocity_field()
        assert np.linalg.norm(l.cons_vel[1, 0][e.active_mask]) <= (
            l.rossby_rate_bound/l.beta*np.linalg.norm(q))


def test_linear_channel_rossby_wave_time_refinement():
    """Small barotropic wave: beta integration versus its discrete frequency."""
    n = 24
    k, m, amplitude, beta = 2*np.pi, np.pi, 1.e-9, 1000.
    dx = 1/n
    lam = 4*(np.sin(k*dx/2)**2+np.sin(m*dx/2)**2)/dx**2
    omega = -beta*np.sin(k*dx)/dx*np.cos(m*dx/2)**2/lam
    T = 4*2*np.pi/abs(omega)
    errors = []
    for cfl in (.15, .075, .0375):
        mesh = grid(n, n)
        mesh.boundary_conditions = {"Channel": (None, (1., 0., 0.), None, (1., 0., 0.))}
        s = qg2(mesh, obstacle=False, beta=beta, nu=0., linear_drag=0., units="dimensionless")
        def psi(x, y, t=0.):
            return amplitude*np.cos(k*x-omega*t)*np.sin(m*y)
        s.set_potential_vorticity_ivp(lambda x, y: np.stack([-lam*psi(x, y)]*2),
            lambda x, y: (np.stack([-amplitude*m*np.cos(k*x)*np.cos(m*y)]*2),
                          np.stack([-amplitude*k*np.sin(k*x)*np.sin(m*y)]*2)))
        s.cfl = cfl
        s.final_time = T
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            s.solve(visual=-1)
        l = s.layers[0]
        exact = psi(l.X[:-1, :-1], l.Y[:-1, :-1], T)
        errors.append(np.linalg.norm((s.psi_c[0]-exact)[s.active_mask])/np.linalg.norm(exact[s.active_mask]))
    assert errors[0] < .04
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5
