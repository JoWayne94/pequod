"""SI inputs/outputs around the dimensionless QG numerical engines."""
from dataclasses import dataclass

import numpy as np

from ..grid import Grid

SECONDS_PER_YEAR = 365.25 * 86400.
DEFAULT_BETA = 2.e-11
DEFAULT_DRAG = 4.e-8
DEFAULT_NU = 100.


@dataclass(frozen=True)
class QGScales:
    length: float = 3_840_000.
    velocity: float = .01

    @property
    def time(self):
        return self.length / self.velocity

    @property
    def streamfunction(self):
        return self.length * self.velocity

    @property
    def beta(self):
        return 1 / (self.length * self.time)

    @property
    def diffusivity(self):
        return self.length * self.velocity


QG_SCALES = QGScales()


def scaled_boundary_conditions(conditions, length, velocity):
    result = {}
    for mode, values in conditions.items():
        if mode == "Periodic":
            # Grid's setter expects no data for periodic boundaries, even
            # though its constructor represents the default with strings.
            result[mode] = None
            continue
        converted = []
        for value in values:
            if value is None:
                converted.append(None)
            elif len(value) == 3:
                # Existing wall tuple: dimensionless selector, slip length, velocity.
                a, slip_length, speed = value
                converted.append((a, slip_length/length, speed/velocity))
            else:
                converted.append(tuple(v/velocity for v in value))
        result[mode] = tuple(converted)
    return result


def scaled_grid(domain, length, velocity):
    mesh = Grid(int(domain.nx), int(domain.ny), domain.west/length,
                domain.south/length, domain.east/length, domain.north/length)
    mesh.boundary_conditions = scaled_boundary_conditions(
        domain.boundary_conditions, length, velocity)
    return mesh


class QGSIInterface:
    """Public SI view; numerical engines operate only on dimensionless data.

    Geometry/callback coordinates use metres, velocities m/s, PV s^-1,
    forcing s^-2, and all solver times seconds. Field reads in SI mode
    return converted snapshots; prescribe state with the IVP methods.
    ``dimensionless`` exposes the engine for numerical diagnostics and raw
    output. ``units='dimensionless'`` retains the benchmark API unchanged.
    """

    def _attach(self, core, domain, units):
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_physical_domain", domain)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "scales", QG_SCALES)
        object.__setattr__(self, "_layer_views", None)
        if units == "SI":
            core._si_interface = self

    @property
    def dimensionless(self):
        return self._core

    @property
    def reference_length(self):
        return self.scales.length

    @property
    def reference_velocity(self):
        return self.scales.velocity

    @property
    def reference_time(self):
        return self.scales.time

    @property
    def domain(self):
        return self._physical_domain

    def _factor(self, name):
        if self.units == "dimensionless":
            return 1.
        s = self.scales
        if name in {"t", "dt", "dt_prev", "final_time", "dt_d"}:
            return s.time
        if name == "beta":
            return s.beta
        if name == "nu":
            return s.diffusivity
        if name in {"linear_drag", "potential_vorticity", "relative_vorticity", "solutions"}:
            return 1/s.time
        if name in {"forcing", "external_forcing"}:
            return 1/s.time**2
        if name in {"coupling", "inverse_deformation_radius_squared"}:
            return 1/s.length**2
        if name in {"psi_c", "psi_v", "_channel_psi_north", "_channel_psi_south", "_obstacle_streamfunction"}:
            return s.streamfunction
        if name in {"cons_vel", "adv_vel", "channel_throughflow"}:
            return s.velocity
        if name in {"deformation_radius", "obstacle_size", "obstacle_width", "obstacle_height",
                    "slip_length",
                    "obstacle_center", "X", "Y", "X_adv_x", "Y_adv_x", "X_adv_y", "Y_adv_y",
                    "x_coords_faces", "y_coords_faces", "points"}:
            return s.length
        return 1.

    def __getattr__(self, name):
        core = object.__getattribute__(self, "_core")
        if name == "layers" and self.units == "SI":
            if self._layer_views is None:
                views = []
                for layer in core.layers:
                    view = QGSIInterface()
                    view._attach(layer, self.domain, "SI")
                    views.append(view)
                object.__setattr__(self, "_layer_views", tuple(views))
            return self._layer_views
        value = getattr(core, name)
        if value is None:
            return None
        factor = self._factor(name)
        if factor == 1:
            return value
        if isinstance(value, tuple):
            return tuple(v*factor for v in value)
        return value*factor

    def __setattr__(self, name, value):
        if name in {"units", "scales", "_core", "_physical_domain", "_layer_views"}:
            raise AttributeError(f"{name} is read-only.")
        factor = self._factor(name)
        if factor != 1 and value is not None:
            value = np.asarray(value)/factor
            if value.ndim == 0:
                value = float(value)
        setattr(self._core, name, value)

    def _sync_boundary_conditions(self):
        if self.units == "SI":
            self._core.domain.boundary_conditions = scaled_boundary_conditions(
                self.domain.boundary_conditions, self.scales.length, self.scales.velocity)

    def _callback(self, callback, factor):
        if callback is None or self.units == "dimensionless":
            return callback
        length = self.scales.length
        def converted(x, y):
            return np.asarray(callback(x*length, y*length))*factor
        return converted

    def set_ivp(self, velocity=None, **kwargs):
        if "ic" in kwargs:
            if velocity is not None:
                raise TypeError("Specify velocity or ic, not both.")
            velocity = kwargs.pop("ic")
        self._sync_boundary_conditions()
        if self.units == "SI" and velocity is not None:
            source = velocity
            def velocity(x, y):
                u, v = source(x*self.scales.length, y*self.scales.length)
                return np.asarray(u)/self.scales.velocity, np.asarray(v)/self.scales.velocity
        return self._core.set_ivp(velocity, **kwargs)

    def set_potential_vorticity_ivp(self, potential_vorticity, velocity):
        if not callable(potential_vorticity) or not callable(velocity):
            raise TypeError("PV and velocity initial conditions must be callable.")
        self.set_ivp(velocity)
        # Reuse the already converted velocity rather than scaling it twice.
        pv = self._callback(potential_vorticity, self.scales.time)
        if hasattr(self._core, "_pv_ic"):
            self._core._pv_ic = pv
        else:
            self._core._prescribed_pv_initial_condition = pv
            self._core._needs_pv_initialisation = True

    def set_forcing_terms(self, forcing=None, *, f=None):
        if f is not None:
            if forcing is not None:
                raise TypeError("Specify forcing or f, not both.")
            forcing = f
        return self._core.set_forcing_terms(self._callback(forcing, self.scales.time**2))

    def get_pre_processes(self):
        self._sync_boundary_conditions()
        return self._core.get_pre_processes()

    def initialize(self):
        self._sync_boundary_conditions()
        if hasattr(self._core, "initialize"):
            return self._core.initialize()
        for process in self._core.get_pre_processes():
            process()

    def step(self):
        if hasattr(self._core, "step"):
            return self._core.step()
        for process in self._core.get_solve_sequence():
            process()
        self._core.t += self._core.dt

    @property
    def velocity(self):
        """Instantaneous central-difference velocity diagnosed from psi."""
        core = self._core
        engines = getattr(core, "layers", (core,))
        us, vs = [], []
        for layer in engines:
            p = layer.psi_c
            west, east = np.roll(p, 1, 1), np.roll(p, -1, 1)
            south, north = np.roll(p, 1, 0), np.roll(p, -1, 0)
            mode = next(iter(layer.domain.boundary_conditions))
            if mode != "Periodic":
                south[0] = 2*p[0]-p[1]
                north[-1] = layer._channel_psi_north if mode == "Channel" else 0.
            if mode == "Basin":
                west[:, 0] = 2*p[:, 0]-p[:, 1]
                east[:, -1] = 0.
            u = -(north-south)/(2*layer.domain.dy)
            v = (east-west)/(2*layer.domain.dx)
            active = layer._active_mask(mode)
            u[~active] = 0.
            v[~active] = 0.
            us.append(u)
            vs.append(v)
        factor = self.scales.velocity if self.units == "SI" else 1.
        if hasattr(core, "layers"):
            return np.stack(us)*factor, np.stack(vs)*factor
        return us[0]*factor, vs[0]*factor

    @property
    def max_velocity_by_layer_m_s(self):
        u, v = self.velocity
        if self.units == "dimensionless":
            u, v = u*self.scales.velocity, v*self.scales.velocity
        speed = np.hypot(u, v)
        return np.atleast_1d(np.max(speed, axis=(-2, -1)))

    @property
    def max_velocity_m_s(self):
        return float(np.max(self.max_velocity_by_layer_m_s))

    def solve(self, save_freq=0, visual=0):
        self._sync_boundary_conditions()
        return self._core.solve(save_freq=save_freq, visual=visual)

    def save_animation(self, *args, **kwargs):
        if hasattr(self._core, "layers"):
            kwargs.setdefault("physical_scales", self.scales if self.units == "SI" else None)
        return self._core.save_animation(*args, **kwargs)
