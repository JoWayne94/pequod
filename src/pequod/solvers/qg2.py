"""Rigid-lid two-layer QG with synchronized CABARET transport.

q1 = lap(psi1) + F1*(psi2-psi1)
q2 = lap(psi2) + F2*(psi1-psi2)
qi_t + J(psii,qi) + beta*psii_x = nu_i*lap(qi) - r_i*lap(psii) + Si.

Both layers share geometry. By default streamfunction boundary data are zero.
Opt-in no-slip channels support independent fixed layer transports. Island
constants are prescribed at the midchannel transport level, not determined
by an island circulation constraint.
"""

from pathlib import Path

import numpy as np

from .qg import _QuasiGeostrophicCore, _MaskedHelmholtzSolver
from .qg_units import (QGSIInterface, QG_SCALES, DEFAULT_BETA, DEFAULT_DRAG,
                       DEFAULT_NU, scaled_grid)
from .incns import IncNavierStokes


class _PeriodicPoisson:
    """Five-point periodic Poisson inverse with an explicit mean constraint."""

    def __init__(self, domain):
        nx, ny = int(domain.nx), int(domain.ny)
        eigenvalues = (
            -4*np.sin(np.pi*np.fft.fftfreq(nx))**2/domain.dx**2
            -4*np.sin(np.pi*np.fft.fftfreq(ny))[:, None]**2/domain.dy**2
        )
        eigenvalues[0, 0] = 1.
        self.inverse = 1/eigenvalues
        self.inverse[0, 0] = 0.

    def solve(self, rhs, out=None):
        if abs(np.mean(rhs)) > 1.e-10 * max(1., np.max(abs(rhs))):
            raise ValueError("Periodic barotropic PV must have zero mean.")
        result = np.fft.ifft2(np.fft.fft2(rhs)*self.inverse).real
        if out is not None:
            out[...] = result
            return out
        return result


class _Layer(_QuasiGeostrophicCore):
    """One transport engine; its PV is not a single-layer Helmholtz RHS."""

    def _build_streamfunction_solver(self, boundary_mode, coordinates):
        # Only the owner performs inversion; reuse its modal factorization.
        return self.owner._barotropic_solver

    def _validate_streamfunction_configuration(self, boundary_mode):
        # The owner supplies the nonhomogeneous coupled boundary lift.
        throughflow = self._channel_throughflow
        try:
            self._channel_throughflow = 0.
            super()._validate_streamfunction_configuration(boundary_mode)
        finally:
            self._channel_throughflow = throughflow
        if (throughflow or self.owner.no_slip) and boundary_mode != "Channel":
            raise ValueError("QG2 throughflow/no-slip requires Channel boundaries.")

    def _psi_boundary_conditions(self):
        super()._psi_boundary_conditions()
        if self.owner.no_slip and self.slip_length is None:
            self.flux_funcs.append(self._channel_wall_pv)

    def _wall_stretching(self, face):
        other = self.owner.layers[1-self.layer_index]
        return self.owner.coupling[self.layer_index]*(other._wall_constant(face)-self._wall_constant(face))

    def _channel_wall_pv(self):
        other = self.owner.layers[1-self.layer_index]
        F = self.owner.coupling[self.layer_index]
        south = 2*(self.psi_c[1]-self._channel_psi_south)/self.domain.dy**2
        north = (2*(self.psi_c[-1]-self._channel_psi_north)/self.domain.dy**2
                 + F*(other._channel_psi_north-self._channel_psi_north))
        self.solutions[0, 0] = south
        self._adv_y[1, 0, 0] = south
        self._adv_x[0, 0, 0] = .5*(south+np.roll(south, -1))
        self._adv_y[0, 0, -1] = north

    def mask_cmm(self):
        if self.slip_length is not None or not self.owner.no_slip:
            return super().mask_cmm()
        self._channel_wall_pv()
        IncNavierStokes.mask_cmm(self)
        if self.has_obstacle:
            other = self.owner.layers[1-self.layer_index]
            boundary = self.obstacle_mask & ~self.obstacle_interior
            self.solutions[0, boundary] += self.owner.coupling[self.layer_index]*(
                other._obstacle_streamfunction-self._obstacle_streamfunction)

    @property
    def relative_vorticity(self):
        other = self.owner.layers[1-self.layer_index]
        return (self.potential_vorticity
                + self.owner.coupling[self.layer_index]*(self.psi_c-other.psi_c))


class _TwoLayerQGCore:
    """Two coupled PV layers on one Grid, with periodic/channel/basin BCs.

    F1,F2 > 0 are stretching coefficients, conventionally f0^2/(g' Hi).
    Scalar or two-element ``nu``/``linear_drag`` values configure each layer.
    Array properties return snapshots with shape (2, ny, nx). ``layers`` exposes the two
    existing QG transport engines for diagnostics, not independent stepping.
    Set parameters and grid BCs before calling ``initialize`` or ``solve``.

    ``no_slip=True`` opts into channel/island Thom wall-vorticity closures,
    including the layer-stretching contribution to boundary PV. In that mode
    ``channel_throughflow=(U1, U2)`` prescribes the layer transports via
    south/north streamfunction values 0 and -Ui*channel_height. The centered
    island streamfunction is fixed at -Ui*(island_y-south).
    """

    def __init__(self, domain, obstacle=True, obstacle_size=None, *,
                 F1=1., F2=1., beta=0., nu=0., linear_drag=(0., 0.),
                 channel_throughflow=(0., 0.), no_slip=False, slip_length=None):
        self._coupling = (float(F1), float(F2))
        if not all(np.isfinite(f) and f > 0 for f in self.coupling):
            raise ValueError("F1 and F2 must be finite and strictly positive.")
        self.domain = domain
        flow = np.broadcast_to(np.asarray(channel_throughflow, dtype=float), (2,))
        if not np.isfinite(flow).all():
            raise ValueError("channel_throughflow must contain finite velocities.")
        self.no_slip = bool(no_slip)
        if self.no_slip and slip_length not in (None, 0.):
            raise ValueError("Use no_slip or slip_length, not conflicting wall conditions.")
        if np.any(flow) and not self.no_slip and slip_length is None:
            raise ValueError("Nonzero QG2 throughflow requires no_slip=True or an explicit slip_length.")
        self.layers = tuple(_Layer(domain, obstacle, obstacle_size,
                                   deformation_radius=np.inf, beta=beta, slip_length=slip_length)
                            for _ in range(2))
        for i, layer in enumerate(self.layers):
            layer.owner, layer.layer_index = self, i
            layer.channel_throughflow = flow[i]
        self.nu = nu
        self.linear_drag = linear_drag
        self.cfl = .15
        self.final_time = 1.
        self.t = 0.
        self.dt = 0.
        self.data_dir = Path("data_dir_qg2")
        self._pv_ic = None
        self._velocity_ic = None
        self._initialized = False
        self._saved_frames = []

    @property
    def coupling(self):
        return self._coupling

    @property
    def beta(self):
        return self.layers[0].beta

    @property
    def slip_length(self):
        return self.layers[0].slip_length

    @slip_length.setter
    def slip_length(self, value):
        if self._initialized:
            raise RuntimeError("Configure slip_length before initialization.")
        for layer in self.layers:
            layer.slip_length = value

    @beta.setter
    def beta(self, value):
        for layer in self.layers:
            layer.beta = value

    @property
    def potential_vorticity(self):
        return np.stack([layer.potential_vorticity for layer in self.layers])

    @property
    def solutions(self):
        return self.potential_vorticity

    @property
    def psi_c(self):
        return np.stack([layer.psi_c for layer in self.layers])

    @property
    def relative_vorticity(self):
        return np.stack([layer.relative_vorticity for layer in self.layers])

    @property
    def cfl(self):
        return self.layers[0].cfl

    @cfl.setter
    def cfl(self, value):
        for layer in self.layers:
            layer.cfl = value

    @property
    def nu(self):
        return tuple(layer.nu for layer in self.layers)

    @nu.setter
    def nu(self, value):
        for layer, coefficient in zip(self.layers, np.broadcast_to(value, (2,))):
            layer.nu = coefficient

    @property
    def linear_drag(self):
        return tuple(layer.linear_drag for layer in self.layers)

    @linear_drag.setter
    def linear_drag(self, value):
        for layer, coefficient in zip(self.layers, np.broadcast_to(value, (2,))):
            layer.linear_drag = coefficient

    def set_ivp(self, velocity):
        """Velocity callback returns (u,v), each with leading layer axis 2."""
        self._velocity_ic = velocity
        self._pv_ic = None
        self._initialized = False

    def set_potential_vorticity_ivp(self, potential_vorticity, velocity):
        """PV callback returns (2,ny,nx); velocity callback returns (u,v)."""
        self.set_ivp(velocity)
        self._pv_ic = potential_vorticity

    def set_forcing_terms(self, forcing=None):
        """Time-independent external sources with shape (2,ny,nx)."""
        x, y = self.layers[0].X[:-1, :-1], self.layers[0].Y[:-1, :-1]
        values = np.zeros((2,)+x.shape) if forcing is None else np.asarray(forcing(x, y))
        if values.shape != (2,)+x.shape or not np.isfinite(values).all():
            raise ValueError("Forcing must be finite with shape (2,ny,nx).")
        for layer, source in zip(self.layers, values):
            layer.external_forcing[...] = source
            layer.forcing[...] = source

    def _build_inverses(self):
        base = self.layers[0]
        mode = next(iter(self.domain.boundary_conditions))
        base._validate_streamfunction_configuration(mode)
        self.active_mask = base._active_mask(mode)
        kwargs = dict(periodic_x=mode in {"Periodic", "Channel"},
                      periodic_y=mode == "Periodic")
        if mode == "Periodic" and not base.has_obstacle:
            self._barotropic_solver = _PeriodicPoisson(self.domain)
        else:
            self._barotropic_solver = _MaskedHelmholtzSolver(
                self.active_mask, self.domain.dx, self.domain.dy, 0., **kwargs)
        self._baroclinic_solver = _MaskedHelmholtzSolver(
            self.active_mask, self.domain.dx, self.domain.dy,
            sum(self.coupling), **kwargs)
        self._boundary_lift = np.zeros((2,)+self.active_mask.shape)
        self._laplace_lift = self._boundary_lift.copy()
        if mode == "Channel":
            # Grid values on excluded south/island nodes plus omitted north row.
            boundaries = np.zeros_like(self._boundary_lift)
            north = np.array([l._channel_psi_north for l in self.layers])
            for i, layer in enumerate(self.layers):
                if layer.has_obstacle:
                    boundaries[i, layer.obstacle_mask] = layer._obstacle_streamfunction

            def lift(b, bn, inverse):
                neighbours = ((np.roll(b, 1, 1)+np.roll(b, -1, 1))/self.domain.dx**2
                              + (np.roll(b, 1, 0)+np.roll(b, -1, 0))/self.domain.dy**2)
                neighbours[-1] += (bn-b[0])/self.domain.dy**2
                return b + inverse.solve(-neighbours)

            F1, F2 = self.coupling
            mean = lift((F2*boundaries[0]+F1*boundaries[1])/(F1+F2),
                        (F2*north[0]+F1*north[1])/(F1+F2), self._barotropic_solver)
            difference = lift(boundaries[0]-boundaries[1], north[0]-north[1],
                              self._baroclinic_solver)
            self._boundary_lift[0] = mean+F1/(F1+F2)*difference
            self._boundary_lift[1] = mean-F2/(F1+F2)*difference
            for i in range(2):
                self._laplace_lift[i] = lift(boundaries[i], north[i], self._barotropic_solver)
        self._has_nonzero_lift = bool(np.any(self._boundary_lift))
        self._mean_rhs = np.empty(self.active_mask.shape)
        self._difference_rhs = np.empty_like(self._mean_rhs)
        self._mean_psi = np.empty_like(self._mean_rhs)
        self._difference_psi = np.empty_like(self._mean_rhs)

    def compute_stream_functions(self):
        """Invert both layers together via weighted barotropic/difference modes."""
        F1, F2 = self.coupling
        q1, q2 = self.layers[0].solutions[0], self.layers[1].solutions[0]
        np.multiply(q1,F2,out=self._mean_rhs)
        np.multiply(q2,F1,out=self._difference_rhs)
        self._mean_rhs += self._difference_rhs
        self._mean_rhs /= F1+F2
        np.subtract(q1,q2,out=self._difference_rhs)
        mean = self._barotropic_solver.solve(self._mean_rhs,out=self._mean_psi)
        difference = self._baroclinic_solver.solve(self._difference_rhs,out=self._difference_psi)
        np.multiply(difference,F1/(F1+F2),out=self.layers[0].psi_c)
        self.layers[0].psi_c += mean
        np.multiply(difference,-F2/(F1+F2),out=self.layers[1].psi_c)
        self.layers[1].psi_c += mean
        for i, layer in enumerate(self.layers):
            if self._has_nonzero_lift:
                layer.psi_c[...] += self._boundary_lift[i]
            layer.psi_v_bcs()

    def initialize(self):
        if self._velocity_ic is None:
            raise ValueError("Set an initial condition before initialization.")
        self._build_inverses()
        for i, layer in enumerate(self.layers):
            def velocity(x, y, i=i):
                u, v = self._velocity_ic(x, y)
                u, v = np.asarray(u), np.asarray(v)
                if (u.shape != (2,)+x.shape or v.shape != (2,)+y.shape
                        or not np.isfinite(u).all() or not np.isfinite(v).all()):
                    raise ValueError("Velocity must return finite u,v with shape (2,ny,nx).")
                return u[i], v[i]
            layer.set_ivp(velocity)
            layer._boundary_conditions()
            layer._laplacian_boundary_conditions()
            layer._psi_boundary_conditions()
            if self._pv_ic is not None:
                def pv(x, y, i=i):
                    values = np.asarray(self._pv_ic(x, y))
                    if values.shape != (2,)+x.shape or not np.isfinite(values).all():
                        raise ValueError("PV must be finite with shape (2,ny,nx).")
                    return values[i]
                layer._prescribed_pv_initial_condition = pv
                layer._initialise_potential_vorticity()
        if self._pv_ic is None:
            zeta = self.potential_vorticity
            psi = np.stack([self._barotropic_solver.solve(z) for z in zeta]) + self._laplace_lift
            for i, layer in enumerate(self.layers):
                layer.solutions[0] = zeta[i] + self.coupling[i]*(psi[1-i]-psi[i])
                # Second-order face initialization from conservative PV.
                layer._adv_x[:, 0] = .5*(layer.solutions[0]+np.roll(layer.solutions[0], -1, 1))
                layer._adv_y[:, 0] = .5*(layer.solutions[0]+np.roll(layer.solutions[0], -1, 0))
        self.compute_stream_functions()
        for layer in self.layers:
            layer.t = self.t = 0.
            layer.dt = 0.
            layer.dt_prev = 0.
            layer.alpha_t = 0.
            layer._Q = np.zeros_like(layer.solutions)
            for name in ("_M_x", "_m_x", "_M_y", "_m_y"):
                setattr(layer, name, np.zeros_like(layer.solutions))
            layer.compute_velocity_field()
            layer.mask_cmm()
        self._initialized = True

    def step(self):
        """Advance both layers through matching predictor/corrector stages."""
        if not self._initialized:
            self.initialize()
        if self.t >= self.final_time:
            return
        candidates = []
        for layer in self.layers:
            layer.final_time = self.final_time
            layer.dt_d = layer.get_diffusion_timestep
            layer.update_dt(adv=layer.update_c, diff=layer.diffusion_dt)
            candidates.append(layer.dt)
        # Each candidate already includes advection, diffusion and the
        # beta/drag source bound; synchronize both transport engines.
        self.dt = min(candidates)
        for layer in self.layers:
            layer.dt = self.dt
            layer.update_alpha()
            layer.predictor()
            layer.numerical_fluxes()
        self.compute_stream_functions()
        for layer in self.layers:
            layer.mask_cmm()
            layer.compute_velocity_field()
            layer.update_maximum_principle()
            layer.second_order_extrapolation()
            layer.numerical_fluxes()
        for layer in self.layers:
            layer.corrector()
        self.compute_stream_functions()
        self.t += self.dt
        for layer in self.layers:
            layer.mask_cmm()
            layer.update_dt_prev()
            layer.t = self.t

    def solve(self, save_freq=0, visual=0):
        """Advance to final_time; optionally save q1,q2,psi1,psi2 as NPZ frames."""
        if not isinstance(save_freq, (int, np.integer)) or save_freq < 0:
            raise ValueError("save_freq must be a non-negative integer.")
        if not np.isfinite(self.final_time) or self.final_time < self.t:
            raise ValueError("final_time must be finite and not precede current time.")
        if not self._initialized:
            self.initialize()
        step = 0
        self._saved_frames = []
        if save_freq:
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        while self.t < self.final_time:
            if save_freq and step % save_freq == 0:
                path = Path(self.data_dir)/f"qg2_{step:06d}.npz"
                np.savez(path,
                         t=self.t, q=self.potential_vorticity, psi=self.psi_c)
                self._saved_frames.append(path)
                if visual != -1 and hasattr(self, "_si_interface"):
                    si = self._si_interface
                    print(f"t={si.t:.6g} s, dt={si.dt:.6g} s, "
                          f"max velocity={si.max_velocity_m_s:.6g} m/s", flush=True)
            self.step()
            step += 1
        if save_freq:
            path = Path(self.data_dir)/f"qg2_{step:06d}.npz"
            np.savez(path,
                     t=self.t, q=self.potential_vorticity, psi=self.psi_c)
            self._saved_frames.append(path)
        if visual != -1:
            if hasattr(self, "_si_interface"):
                si = self._si_interface
                print(f"Two-layer QG completed: t={si.t:g} s, steps={step}, "
                      f"max velocity={si.max_velocity_m_s:.6g} m/s")
            else:
                print(f"Two-layer QG completed: t={self.t:g}, steps={step}")

    def save_animation(self, ani_name="qg2.gif", *, fps=12, dpi=100, physical_scales=None):
        """Save the latest solve's frames as a 2x2 GIF; retain NPZ data.

        Rows are layers; columns are streamfunction and PV. Color scales
        are fixed over time and shared by both layers for each variable.
        Call solve(save_freq=...) first. Only frames from that call are used.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Rectangle

        if not self._saved_frames:
            raise ValueError("Call solve with save_freq > 0 before save_animation.")
        output = Path(ani_name)
        if output.suffix.lower() != ".gif":
            raise ValueError("Animation output must have a .gif extension.")
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(dpi) or dpi <= 0:
            raise ValueError("fps and dpi must be finite and positive.")
        d = self.domain
        mode = next(iter(d.boundary_conditions))
        layer = self.layers[0]
        length_scale = physical_scales.length if physical_scales else 1.
        field_scale = {"psi": physical_scales.streamfunction if physical_scales else 1.,
                       "q": 1/physical_scales.time if physical_scales else 1.}
        limits = {"psi": 0., "q": 0.}
        expected_shape = (2, int(d.ny), int(d.nx))
        for path in self._saved_frames:
            with np.load(path) as frame:
                for field in limits:
                    data = frame[field]
                    if data.shape != expected_shape or not np.isfinite(data).all():
                        raise ValueError(f"Invalid {field} data in {path}.")
                    limits[field] = max(limits[field], np.max(abs(data[:, self.active_mask])))

        def plot_data(data, i, field, frame):
            # Add the omitted north/east endpoints, without shifting rows.
            full = np.pad(data, ((0, 1), (0, 1)))
            if mode in {"Periodic", "Channel"}:
                full[:-1, -1] = data[:, 0]
            if mode == "Periodic":
                full[-1, :-1] = data[0, :]
                full[-1, -1] = data[0, 0]
            elif mode == "Channel":
                if field == "psi":
                    full[-1] = self.layers[i]._channel_psi_north
                elif self.no_slip:
                    cn = self.layers[i]._channel_psi_north
                    other = self.layers[1-i]._channel_psi_north
                    full[-1, :-1] = (2*(frame["psi"][i, -1]-cn)/d.dy**2
                                     + self.coupling[i]*(other-cn))
                    full[-1, -1] = full[-1, 0]
            solid = np.pad(layer.obstacle_mask if layer.has_obstacle
                           else np.zeros_like(data, dtype=bool), ((0, 1), (0, 1)))
            if layer.obstacle_kind == "northwest_bump":
                solid[-1, :-1] = layer.obstacle_mask[-1]
            elif layer.obstacle_kind == "southeast_bump":
                solid[:-1, -1] = layer.obstacle_mask[:, -1]
            return np.ma.array(full*field_scale[field], mask=solid)

        output.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans",
                             "font.size": 10}):
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
            artists = []
            try:
                with np.load(self._saved_frames[0]) as first:
                    for i in range(2):
                        for j, field in enumerate(("psi", "q")):
                            ax = axes[i, j]
                            bound = (limits[field] or 1.)*field_scale[field]

                            if field == "q":
                                bound *= 0.25

                            artist = ax.imshow(
                                plot_data(first[field][i], i, field, first), origin="lower",
                                extent=tuple(length_scale*z for z in
                                             (d.west-d.dx/2, d.east+d.dx/2,
                                              d.south-d.dy/2, d.north+d.dy/2)),
                                cmap="RdBu_r", vmin=-bound, vmax=bound,
                                interpolation="nearest")
                            ax.set(xlim=(d.west*length_scale, d.east*length_scale),
                                   ylim=(d.south*length_scale, d.north*length_scale),
                                   xlabel="x (m)" if physical_scales else "x",
                                   ylabel="y (m)" if physical_scales else "y",
                                   title=f"Layer {i+1}: " + ("streamfunction" if field == "psi" else "PV anomaly"))
                            if layer.has_obstacle:
                                x0, y0 = layer.obstacle_center
                                ax.add_patch(Rectangle(
                                    ((x0-layer.obstacle_width/2)*length_scale,
                                     (y0-layer.obstacle_height/2)*length_scale),
                                    layer.obstacle_width*length_scale, layer.obstacle_height*length_scale,
                                    facecolor="0.65", edgecolor="black", linewidth=.8))
                            label = rf"$\psi_{i+1}$" if field == "psi" else rf"$q_{i+1}$"
                            if physical_scales:
                                label += " (m²/s)" if field == "psi" else " (s⁻¹)"
                            fig.colorbar(artist, ax=ax, label=label)
                            artists.append(artist)
                title = fig.suptitle("")

                def update(index):
                    with np.load(self._saved_frames[index]) as frame:
                        for artist, (i, field) in zip(artists, ((0, "psi"), (0, "q"), (1, "psi"), (1, "q"))):
                            artist.set_data(plot_data(frame[field][i], i, field, frame))
                        time_label = f"{float(frame['t']):.5g}"
                        if physical_scales:
                            from .qg_units import SECONDS_PER_YEAR
                            time_label = f"{float(frame['t'])*physical_scales.time/SECONDS_PER_YEAR:.5g} years"
                        title.set_text(f"Two-layer QG | {mode} | t = {time_label}")
                    return (*artists, title)

                animation = FuncAnimation(fig, update, frames=len(self._saved_frames),
                                          interval=1000/fps, blit=False)
                animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
            finally:
                plt.close(fig)
        return output


class TwoLayerQG(QGSIInterface):
    """Two-layer QG with SI inputs/outputs; solver times are seconds.

    F1/F2 are in m^-2, geometry in metres, velocities in m/s, PV in s^-1,
    sources in s^-2. Default lower-layer drag is 4e-8 s^-1; upper drag is
    zero. Use units='dimensionless' for the existing analytic benchmarks.
    slip_length (metres) applies Navier slip to all stationary outer and
    island/bump faces in both layers. 0=no slip, np.inf=free slip, while
    None retains the legacy benchmark treatment. Configure before initialize.
    """

    def __init__(self, domain, obstacle=True, obstacle_size=None, *,
                 F1=None, F2=None, beta=None, nu=None, linear_drag=None,
                 channel_throughflow=(0., 0.), no_slip=False, slip_length=None, units="SI"):
        if units not in {"SI", "dimensionless"}:
            raise ValueError("units must be 'SI' or 'dimensionless'.")
        si = units == "SI"
        s = QG_SCALES
        L, U, time = (s.length, s.velocity, s.time) if si else (1., 1., 1.)
        mesh = scaled_grid(domain, L, U) if si else domain
        F1 = 1/L**2 if F1 is None else F1
        F2 = 1/L**2 if F2 is None else F2
        beta = (DEFAULT_BETA if si else 0.) if beta is None else beta
        nu = (DEFAULT_NU if si else 0.) if nu is None else nu
        drag = ((0., DEFAULT_DRAG) if si else (0., 0.)) if linear_drag is None else linear_drag
        core = _TwoLayerQGCore(mesh, obstacle,
            None if obstacle_size is None else tuple(np.asarray(obstacle_size)/L),
            F1=F1*L**2, F2=F2*L**2, beta=beta*L*time,
            nu=np.asarray(nu)/(L*U), linear_drag=np.asarray(drag)*time,
            channel_throughflow=np.asarray(channel_throughflow)/U, no_slip=no_slip,
            slip_length=None if slip_length is None else slip_length/L)
        self._attach(core, domain, units)
