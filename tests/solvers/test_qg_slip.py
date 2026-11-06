"""Navier wall physics, optimized reconstruction and finite-CFL evolution."""
import numpy as np
import pytest

from pequod import grid, qg, qg2
from pequod.solvers.incns import IncNavierStokes
from pequod.solvers.qg_units import QG_SCALES, SECONDS_PER_YEAR
from tutorials.qg2.wind_experiment import make_wind_solver, wind_source_amplitude


def zeros(x,y):
    return np.zeros_like(x),np.zeros_like(y)


@pytest.mark.parametrize("factory",[qg,qg2])
@pytest.mark.parametrize("obstacle",["centered","northwest_bump","southeast_bump"])
@pytest.mark.parametrize("ell",[0.,.04,np.inf])
def test_navier_quadratic_normal_profile_all_faces(factory,obstacle,ell):
    mesh=grid(32,32)
    mesh.boundary_conditions={"Basin":((1.,0.,0.),)*4}
    solver=factory(mesh,obstacle=obstacle,obstacle_size=(.125,.125),
                   slip_length=ell,beta=0.,nu=0.,linear_drag=0.,units="dimensionless")
    if factory is qg2:
        solver.set_potential_vorticity_ivp(lambda x,y:np.zeros((2,)+x.shape),
            lambda x,y:(np.zeros((2,)+x.shape),np.zeros((2,)+y.shape)))
    else:
        solver.set_potential_vorticity_ivp(lambda x,y:np.zeros_like(x),zeros)
    solver.initialize()
    engines=solver.dimensionless.layers if factory is qg2 else (solver.dimensionless,)
    for k,layer in enumerate(engines):
        zeta=2.*(k+1)
        h=mesh.dx
        # Locally psi(n)=zeta*(ell*n+n^2/2) satisfies u_t=ell*d_n u_t
        # and has exact wall curvature zeta. Infinity tests the linear limit.
        value=1. if np.isinf(ell) else zeta*(ell*h+.5*h*h)
        active=layer._active_mask("Basin")
        layer.psi_c.fill(0.)
        layer.psi_c[active]=value
        layer.mask_cmm()
        expected=0. if np.isinf(ell) else zeta
        np.testing.assert_allclose([layer.potential_vorticity[0,16],
            layer.potential_vorticity[16,0],layer._wall_q_north[16],
            layer._wall_q_east[16]],expected)
        boundary=layer.obstacle_mask & ~layer.obstacle_interior
        np.testing.assert_allclose(layer.potential_vorticity[boundary],expected)
        layer.u[1:-1,1:-1]=layer.potential_vorticity
        layer.laplacian_bcs()
        assert layer.u[-1,17]==pytest.approx(expected)
        assert layer.u[17,-1]==pytest.approx(expected)
        fx,fy=layer.physical_flux()
        assert np.max(abs(fx[:,layer.obstacle_faces_x]))==0.
        assert np.max(abs(fy[:,layer.obstacle_faces_y]))==0.


def test_partial_slip_pv_includes_boundary_layer_stretching():
    mesh=grid(32,32)
    mesh.boundary_conditions={"Channel":(None,(1.,0.,0.),None,(1.,0.,0.))}
    ell=.03
    solver=qg2(mesh,obstacle="centered",obstacle_size=(.125,.125),
                F1=2.,F2=3.,beta=0.,nu=0.,linear_drag=0.,
                channel_throughflow=(1.,.5),slip_length=ell,units="dimensionless")
    solver.set_potential_vorticity_ivp(lambda x,y:np.zeros((2,)+x.shape),
        lambda x,y:(np.zeros((2,)+x.shape),np.zeros((2,)+y.shape)))
    solver.initialize()
    for i,layer in enumerate(solver.layers):
        zeta=1.7
        value=zeta*(ell*mesh.dy+.5*mesh.dy**2)
        layer.psi_c[1,:]=value
        layer.psi_c[-1,:]=layer._channel_psi_north+value
        layer.mask_cmm()
        Ui,Uj=(1.,.5)[i],(1.,.5)[1-i]
        np.testing.assert_allclose(layer._wall_q_north,zeta+solver.coupling[i]*(Ui-Uj))
        np.testing.assert_allclose(layer.potential_vorticity[0],zeta)


@pytest.mark.parametrize("mode",["Periodic","Channel","Basin"])
def test_optimized_velocity_reconstruction_matches_reference(mode):
    owners=[]
    for _ in range(2):
        mesh=grid(16,16)
        if mode!="Periodic":
            mesh.boundary_conditions={mode:(None,(1.,0.,0.),None,(1.,0.,0.))
                if mode=="Channel" else ((1.,0.,0.),)*4}
        solver=qg2(mesh,obstacle=False,beta=.3,nu=.01,linear_drag=.02,units="dimensionless")
        solver.set_ivp(lambda x,y:(np.zeros((2,)+x.shape),np.zeros((2,)+y.shape)))
        solver.initialize()
        owners.append(solver.dimensionless)
    actual,reference=[owner.layers[0] for owner in owners]
    rng=np.random.default_rng(44)
    for name in ("psi_v","psi_c","external_forcing","_cons","_adv_x","_adv_y",
                 "_Q","_M_x","_M_y","_m_x","_m_y"):
        values=rng.normal(size=getattr(actual,name).shape)
        getattr(actual,name)[...]=values
        getattr(reference,name)[...]=values
    actual.dt=reference.dt=.01
    actual.alpha_t=reference.alpha_t=.4
    for _ in range(3):
        actual.compute_velocity_field()
        IncNavierStokes.compute_velocity_field(reference)
        for name in ("adv_vel","adv_vel_raw_prev","cons_vel","_M_x","_M_y","_m_x","_m_y"):
            np.testing.assert_allclose(getattr(actual,name),getattr(reference,name),atol=1.e-12)


def test_physical_partial_slip_wind_cfl_04_stays_finite():
    s=make_wind_solver(32,QG_SCALES.length*6/32,wind_source_amplitude(),
                       bump_cells=4,slip_length=30000.,cfl=.4)
    assert s.slip_length==30000.
    assert s.dimensionless.slip_length==pytest.approx(30000/QG_SCALES.length)
    s.initialize()
    s.final_time=.01*SECONDS_PER_YEAR
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        while s.dimensionless.t<s.dimensionless.final_time:
            s.step()
    assert np.isfinite(s.potential_vorticity).all()
    assert 0<s.max_velocity_m_s<.1
    for layer in s.dimensionless.layers:
        np.testing.assert_allclose(layer.psi_c[~s.dimensionless.active_mask],0.)


def test_partial_slip_manufactured_basin_at_advective_cfl_04():
    from tutorials.qg2.main import make_case,exact_fields
    case="basin_northwest_manufactured"
    solver,_=make_case(32,case)
    solver.beta=0.
    solver.slip_length=.04
    solver.cfl=.4
    solver.set_forcing_terms(lambda x,y:exact_fields(x,y,case,beta=0.)[4])
    solver.final_time=.5
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        solver.solve(visual=-1)
    layer=solver.layers[0]
    reference=exact_fields(layer.X[:-1,:-1],layer.Y[:-1,:-1],case,beta=0.)[0]
    for actual,expected in zip(solver.psi_c,reference):
        assert np.linalg.norm((actual-expected)[solver.active_mask])/np.linalg.norm(expected[solver.active_mask])<.05


def test_obstacle_face_next_to_omitted_regular_boundary_node():
    mesh=grid(32,32)
    mesh.boundary_conditions={"Basin":((1.,0.,0.),)*4}
    solver=qg(mesh,obstacle="northwest_bump",obstacle_size=(31/32,4/32),
               slip_length=.04,units="dimensionless")
    solver.set_ivp(zeros)
    solver.initialize()
    solver.dimensionless.psi_c[solver._active_mask("Basin")]=.1
    solver.mask_cmm()
    # The east neighbour is the omitted zero-psi regular east boundary.
    assert solver.potential_vorticity[30,31]==0.


@pytest.mark.parametrize("ell",[-1.,np.nan,-np.inf])
def test_invalid_slip_length_rejected(ell):
    with pytest.raises(ValueError,match="slip_length"):
        qg2(grid(16,16),obstacle=False,slip_length=ell,units="dimensionless")
