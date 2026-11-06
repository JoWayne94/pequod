"""Single-layer quasi-geostrophic potential-vorticity solver."""

from typing import Callable

import numpy as np
from scipy.fft import dstn, idstn
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .incns import IncNavierStokes
from .adr import Grid
from . import _qg_kernels as _kernels
from ..helmsolve.helmsolve import PoissonSolver
from .qg_units import (QGSIInterface, QG_SCALES, DEFAULT_BETA, DEFAULT_DRAG,
                       DEFAULT_NU, scaled_grid)


def _shift_into(source, out, axis, step):
    """Unit periodic shift without np.roll's allocation/axis dispatch."""
    if axis == -1:
        if step == 1:
            out[..., 1:] = source[..., :-1]
            out[..., 0] = source[..., -1]
        else:
            out[..., :-1] = source[..., 1:]
            out[..., -1] = source[..., 0]
    elif step == 1:
        out[..., 1:, :] = source[..., :-1, :]
        out[..., 0, :] = source[..., -1, :]
    else:
        out[..., :-1, :] = source[..., 1:, :]
        out[..., -1, :] = source[..., 0, :]


class _MaskedHelmholtzSolver:
    """Invert a grid-aligned Helmholtz operator with optional periodicity."""

    def __init__(
        self,
        active_mask: np.ndarray,
        dx: float,
        dy: float,
        inverse_deformation_radius_squared: float,
        *,
        periodic_x: bool,
        periodic_y: bool,
        backend: str = "auto",
    ) -> None:
        self.active_mask = np.asarray(active_mask, dtype=bool)
        if self.active_mask.ndim != 2:
            raise ValueError("active_mask must be two-dimensional.")
        self.shape = self.active_mask.shape
        self.dx = float(dx)
        self.dy = float(dy)
        self.inverse_deformation_radius_squared = float(
            inverse_deformation_radius_squared
        )
        self.periodic_x = bool(periodic_x)
        self.periodic_y = bool(periodic_y)
        if backend not in {"auto", "sparse"}:
            raise ValueError("Elliptic backend must be 'auto' or 'sparse'.")
        self.backend = "sparse"

        count = np.count_nonzero(self.active_mask)
        if count == 0:
            raise ValueError("The QG domain contains no active fluid points.")
        if (
            self.inverse_deformation_radius_squared == 0.0
            and count == self.active_mask.size
            and self.periodic_x
            and self.periodic_y
        ):
            raise ValueError(
                "The fully periodic Poisson operator needs a zero-mean "
                "spectral inverse."
            )

        # A rectangular basin is diagonal in a DST-I basis. A small solid
        # can be imposed exactly by a capacitance matrix on its exposed
        # nodes, avoiding large sparse triangular solves at every stage.
        if backend == "auto" and self._build_basin_inverse():
            self.backend = "dst-capacitance" if len(self._dst_boundary) else "dst"
            return

        indices = -np.ones(self.shape, dtype=int)
        indices[self.active_mask] = np.arange(count)
        self._indices = indices

        rows = []
        columns = []
        values = []
        diagonal = (
            -2.0 / self.dx**2
            - 2.0 / self.dy**2
            - self.inverse_deformation_radius_squared
        )
        neighbours = (
            (-1, 0, 1.0 / self.dy**2),
            (1, 0, 1.0 / self.dy**2),
            (0, -1, 1.0 / self.dx**2),
            (0, 1, 1.0 / self.dx**2),
        )
        ny, nx = self.shape

        for j, i in np.argwhere(self.active_mask):
            row = indices[j, i]
            rows.append(row)
            columns.append(row)
            values.append(diagonal)

            for dj, di, coefficient in neighbours:
                jj, ii = j + dj, i + di
                if self.periodic_y:
                    jj %= ny
                if self.periodic_x:
                    ii %= nx
                if not (0 <= jj < ny and 0 <= ii < nx):
                    continue
                if self.active_mask[jj, ii]:
                    rows.append(row)
                    columns.append(indices[jj, ii])
                    values.append(coefficient)

        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(count, count),
        ).tocsc()
        # The stencil is symmetric. A symmetric minimum-degree ordering
        # substantially reduces factor fill and triangular-solve work.
        self._factor = splu(matrix, permc_spec="MMD_AT_PLUS_A")
        self._active_indices = np.flatnonzero(self.active_mask)

    def _build_basin_inverse(self):
        if (self.periodic_x or self.periodic_y or min(self.shape) < 3
                or self.active_mask[0].any() or self.active_mask[:, 0].any()):
            return False
        fluid = self.active_mask[1:, 1:]
        solid = ~fluid
        adjacent = np.zeros_like(fluid)
        adjacent[1:] |= fluid[:-1]
        adjacent[:-1] |= fluid[1:]
        adjacent[:, 1:] |= fluid[:, :-1]
        adjacent[:, :-1] |= fluid[:, 1:]
        boundary = np.flatnonzero(solid & adjacent)
        # Large islands retain the general sparse backend. Bound both setup
        # work and cached Green-function memory, including on refined grids.
        if len(boundary) > 64 or len(boundary)*fluid.size*8 > 16*1024**2:
            return False
        ny, nx = fluid.shape
        eigenvalues = (-4*np.sin(np.pi*np.arange(1, nx+1)/(2*(nx+1)))**2/self.dx**2
                       -4*np.sin(np.pi*np.arange(1, ny+1)[:, None]/(2*(ny+1)))**2/self.dy**2
                       -self.inverse_deformation_radius_squared)
        self._dst_inverse = 1/eigenvalues
        self._dst_solid = np.flatnonzero(solid)
        self._dst_boundary = boundary
        self._dst_rhs = np.empty(fluid.shape)
        if len(boundary):
            green = np.empty((len(boundary), ny, nx))
            impulse = np.zeros_like(self._dst_rhs)
            for k, index in enumerate(boundary):
                impulse.fill(0.)
                impulse.ravel()[index] = 1.
                green[k] = self._dst_solve(impulse)
            # -G is SPD, allowing a small cached Cholesky factorization.
            matrix = -green.reshape(len(boundary), -1)[:, boundary].T
            self._capacitance_factor = cho_factor(matrix, check_finite=False)
            self._dst_green = green
        return True

    def _dst_solve(self, rhs):
        coefficients = dstn(rhs, type=1, norm="ortho", workers=1)
        coefficients *= self._dst_inverse
        return idstn(coefficients, type=1, norm="ortho", overwrite_x=True, workers=1)

    def solve(self, rhs: np.ndarray, out=None) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != self.shape:
            raise ValueError(
                f"Expected QG RHS shape {self.shape}, got {rhs.shape}."
            )
        solution = np.zeros(self.shape, dtype=np.float64) if out is None else out
        if solution.shape != self.shape:
            raise ValueError("Output shape must match the elliptic grid.")
        if self.backend in {"dst", "dst-capacitance"}:
            # Copy before clearing out: solve(rhs, out=rhs) is supported too.
            self._dst_rhs[...] = rhs[1:, 1:]
            self._dst_rhs.ravel()[self._dst_solid] = 0.
            interior = self._dst_solve(self._dst_rhs)
            if len(self._dst_boundary):
                weights = cho_solve(self._capacitance_factor,
                                    interior.ravel()[self._dst_boundary], check_finite=False)
                interior += np.einsum("i,ijk->jk", weights, self._dst_green, optimize=False)
            interior.ravel()[self._dst_solid] = 0.
            solution.fill(0.)
            solution[1:, 1:] = interior
            return solution
        active_rhs = rhs.ravel()[self._active_indices]
        solution.fill(0.)
        solution.ravel()[self._active_indices] = self._factor.solve(active_rhs)
        return solution


class _QuasiGeostrophicCore(IncNavierStokes):
    r"""Single-layer QG solver in potential-vorticity form.

    The prognostic variable is the PV anomaly

    ``q = Laplacian(psi) - psi / deformation_radius**2``

    and it evolves according to

    ``q_t + u q_x + v q_y + beta v = nu Laplacian(q) + forcing``,

    with ``u=-psi_y`` and ``v=psi_x``.  The CABARET transport, viscous
    update, velocity reconstruction, output, and obstacle geometry are
    inherited from :class:`IncNavierStokes`; QG-specific wall hooks impose
    impermeable free-slip boundaries without a no-slip vorticity closure.
    """

    def __init__(
        self,
        domain: Grid,
        obstacle: bool | str = True,
        obstacle_size: tuple[float, float] | None = None,
        *,
        deformation_radius: float = 1.0,
        beta: float = 0.0,
        linear_drag: float = 0.0,
        slip_length: float | None = None,
    ) -> None:
        northwest = isinstance(obstacle, str) and obstacle.lower().replace(
            "-", "_"
        ) in {"northwest_bump", "northwest", "nw_bump"}
        super().__init__(
            domain,
            obstacle=False if northwest else obstacle,
            obstacle_size=obstacle_size,
        )
        if northwest:
            self._configure_northwest_bump(obstacle_size)
        self._deformation_radius = 1.0
        self._beta = 0.0
        self._linear_drag = 0.0
        self.deformation_radius = deformation_radius
        self.beta = beta
        self.linear_drag = linear_drag
        self.external_forcing = np.zeros_like(self.forcing)
        self._needs_pv_initialisation = False
        self._prescribed_pv_initial_condition = None
        self._relative_vorticity_solver = None
        self.slip_length = slip_length
        self._wall_cache = None
        self._mask_indices = None
        self._rossby_length = None

    def _configure_northwest_bump(self, size):
        """Build a solid attached to west and north, without periodic wrap."""
        d = self.domain
        width, height = size or (
            0.25 * (d.east - d.west), 0.25 * (d.north - d.south)
        )
        if not (0 < width < d.east - d.west and
                0 < height < d.north - d.south):
            raise ValueError("Bump dimensions must lie strictly inside the basin.")
        cells = np.array([width / d.dx, height / d.dy])
        if not np.allclose(cells, np.rint(cells), atol=1.e-12, rtol=0):
            raise ValueError("Northwest bump edges must align with grid points.")
        i, j = int(round(cells[0])), int(d.ny) - int(round(cells[1]))
        self.obstacle_kind = "northwest_bump"
        self.has_obstacle = True
        self.obstacle_width, self.obstacle_height = width, height
        self.obstacle_center = np.array([d.west + width / 2, d.north - height / 2])
        self.obstacle_mask[...] = False
        self.obstacle_mask[j:, :i + 1] = True
        boundary = np.zeros_like(self.obstacle_mask)
        boundary[j:, i] = True
        boundary[j, 1:i + 1] = True
        self.boundary_indices = np.argwhere(boundary)
        self.obstacle_interior = self.obstacle_mask & ~boundary
        self.points = np.stack((self.X[:-1, :-1][boundary],
                                self.Y[:-1, :-1][boundary]), axis=-1)
        self.west_bi = self.north_bi = np.empty((0, 2), dtype=int)
        self.east_bi = np.column_stack((np.arange(j, int(d.ny)),
                                       np.full(int(d.ny) - j, i)))
        self.south_bi = np.column_stack((np.full(i, j), np.arange(1, i + 1)))
        self.obstacle_faces_x[...] = self.obstacle_mask
        self.obstacle_faces_y[...] = self.obstacle_mask
        self.obstacle_faces_x[:, :-1] |= self.obstacle_mask[:, 1:]
        self.obstacle_faces_y[:-1, :] |= self.obstacle_mask[1:, :]

    def sol_plt(self, solution):
        result = super().sol_plt(solution)
        if self.obstacle_kind == "northwest_bump":
            # North endpoint is omitted from storage, but belongs to the solid.
            result.mask[-1, :-1] = self.obstacle_mask[-1, :]
        return result

    @property
    def deformation_radius(self) -> float:
        return self._deformation_radius

    @deformation_radius.setter
    def deformation_radius(self, value: float) -> None:
        value = float(value)
        if not (np.isinf(value) or (np.isfinite(value) and value > 0.0)):
            raise ValueError(
                "deformation_radius must be positive or np.inf."
            )
        self._deformation_radius = value

    @property
    def inverse_deformation_radius_squared(self) -> float:
        if np.isinf(self.deformation_radius):
            return 0.0
        return 1.0 / self.deformation_radius**2

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("beta must be finite.")
        self._beta = float(value)

    @property
    def linear_drag(self) -> float:
        return self._linear_drag

    @linear_drag.setter
    def linear_drag(self, value: float) -> None:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("linear_drag must be finite and non-negative.")
        self._linear_drag = float(value)

    @property
    def potential_vorticity(self) -> np.ndarray:
        return self.solutions[0]

    @property
    def relative_vorticity(self) -> np.ndarray:
        return (
            self.potential_vorticity
            + self.inverse_deformation_radius_squared * self.psi_c
        )

    def set_ivp(
        self,
        ic: Callable = None,
        var: int = 0,
        visual: int = 0,
    ) -> None:
        """Set a velocity IVP and convert its curl to consistent QG PV."""
        self._prescribed_pv_initial_condition = None
        super().set_ivp(ic=ic, var=var, visual=visual)
        if ic is not None:
            self._needs_pv_initialisation = True

    def set_potential_vorticity_ivp(
        self,
        potential_vorticity: Callable,
        velocity: Callable,
    ) -> None:
        """Set an analytic PV IVP together with its velocity field.

        ``potential_vorticity`` and ``velocity`` must accept NumPy arrays.
        The velocity supplies the initial face velocities used by the wall
        setup; the prescribed PV is evaluated independently at conservative
        points and at both sets of CABARET faces.  This is useful for
        manufactured-solution tests in which reconstructing PV from a
        sampled velocity would introduce an avoidable initialization error.
        """
        if not callable(potential_vorticity):
            raise TypeError("potential_vorticity must be callable.")
        if not callable(velocity):
            raise TypeError("velocity must be callable.")

        self.set_ivp(velocity)
        self._prescribed_pv_initial_condition = potential_vorticity
        self._needs_pv_initialisation = True

    def set_forcing_terms(self, f: Callable = None) -> None:
        """Set the time-independent part of the QG PV forcing."""
        self.forcing[...] = 0.0
        super().set_forcing_terms(f)
        self.external_forcing[...] = self.forcing

    def _validate_streamfunction_configuration(
        self,
        boundary_mode: str,
    ) -> None:
        if boundary_mode not in {"Periodic", "Channel", "Basin"}:
            raise ValueError(f"Unsupported QG boundary mode: {boundary_mode}.")
        if self.slip_length is not None and boundary_mode != "Periodic":
            for bc in self.domain.boundary_conditions[boundary_mode]:
                if bc is not None and len(bc)==3 and bc[2] != 0:
                    raise NotImplementedError("Navier slip currently requires stationary lateral walls.")
        if self.obstacle_kind in {"southeast_bump", "northwest_bump"} and boundary_mode != "Basin":
            raise NotImplementedError(
                "Corner bumps require Basin boundary conditions."
            )
        if self.channel_throughflow != 0.0:
            raise NotImplementedError(
                "QG channel_throughflow with non-homogeneous Helmholtz "
                "boundary data is not implemented."
            )

    def _active_mask(self, boundary_mode: str) -> np.ndarray:
        active = np.ones(
            (self.domain.ny, self.domain.nx),
            dtype=bool,
        )
        if self.has_obstacle:
            active &= ~self.obstacle_mask
        if boundary_mode in {"Channel", "Basin"}:
            active[0, :] = False
        if boundary_mode == "Basin":
            active[:, 0] = False
        return active

    @property
    def slip_length(self):
        return self._slip_length

    @slip_length.setter
    def slip_length(self, value):
        if value is not None and (np.isnan(value) or value < 0):
            raise ValueError("slip_length must be nonnegative, np.inf, or None.")
        self._slip_length = None if value is None else float(value)
        self._wall_cache = None
        self._partial_wall_state = None

    def _cached_mask_indices(self):
        if self._mask_indices is None:
            self._mask_indices = tuple(np.flatnonzero(mask) for mask in (
                self.obstacle_faces_x, self.obstacle_faces_y, self.obstacle_mask))
        return self._mask_indices

    def _wall_constant(self, face):
        if face == "obstacle":
            return self._obstacle_streamfunction
        if face == "north" and next(iter(self.domain.boundary_conditions)) == "Channel":
            return self._channel_psi_north
        return 0.

    def _wall_stretching(self, face):
        return -self.inverse_deformation_radius_squared*self._wall_constant(face)

    def _wall_coefficient(self, h):
        return 0. if np.isinf(self.slip_length) else 2/(h*(h+2*self.slip_length))

    def _obstacle_wall_stencil(self):
        if self._wall_cache is None:
            nx = int(self.domain.nx)
            ny = int(self.domain.ny)
            mode = next(iter(self.domain.boundary_conditions))
            walls, neighbours, coefficients, ghost, ghost_value = [], [], [], [], []
            for indices, dj, di, h in ((self.west_bi,0,-1,self.domain.dx),
                                      (self.east_bi,0,1,self.domain.dx),
                                      (self.south_bi,-1,0,self.domain.dy),
                                      (self.north_bi,1,0,self.domain.dy)):
                if len(indices):
                    j,i = indices.T
                    jj,ii = j+dj,i+di
                    if mode in {"Periodic","Channel"}:
                        ii %= nx
                    if mode == "Periodic":
                        jj %= ny
                    outside = (jj>=ny)|(ii>=nx)|(jj<0)|(ii<0)
                    values = np.zeros(len(indices))
                    values[jj>=ny] = self._wall_constant("north")
                    values[ii>=nx] = self._wall_constant("east")
                    walls.extend(j*nx+i)
                    neighbours.extend(np.where(outside,0,jj*nx+ii))
                    coefficients.extend(np.full(len(indices), self._wall_coefficient(h)))
                    ghost.extend(outside)
                    ghost_value.extend(values)
            wall, inverse = np.unique(walls, return_inverse=True)
            self._wall_cache = (wall.astype(int), np.asarray(neighbours,dtype=int),
                                np.asarray(coefficients), inverse, np.bincount(inverse),
                                np.flatnonzero(ghost), np.asarray(ghost_value))
        return self._wall_cache

    def _apply_partial_slip(self):
        """Stationary Navier walls: u_t=ell*d_n(u_t), n into fluid.

        On a straight grid-aligned face, zeta_w=2(psi_adj-C)/[h(h+2ell)].
        Add boundary stretching to obtain PV, and average adjoining faces
        at obstacle corners. Normal PV flux remains closed separately.
        """
        mode = next(iter(self.domain.boundary_conditions))
        p = self.psi_c
        state = {}
        cx, cy = self._wall_coefficient(self.domain.dx), self._wall_coefficient(self.domain.dy)
        if mode in {"Basin", "Channel"}:
            south = cy*(p[1]-self._wall_constant("south"))+self._wall_stretching("south")
            north = cy*(p[-1]-self._wall_constant("north"))+self._wall_stretching("north")
            state.update(south=south,north=north,south_tangent=.5*(south+np.roll(south,-1)))
        if mode == "Basin":
            west = cx*(p[:,1]-self._wall_constant("west"))+self._wall_stretching("west")
            east = cx*(p[:,-1]-self._wall_constant("east"))+self._wall_stretching("east")
            state.update(west=west,east=east,west_tangent=.5*(west+np.roll(west,-1)))
        if self.has_obstacle:
            wall, adjacent, coefficient, inverse, count, ghost, ghost_value = self._obstacle_wall_stencil()
            if len(wall):
                adjacent_values = p.ravel()[adjacent]
                adjacent_values[ghost] = ghost_value[ghost]
                zeta = np.bincount(inverse, weights=coefficient*(adjacent_values-
                                    self._wall_constant("obstacle")))/count
                state.update(island_indices=wall,island_q=zeta+self._wall_stretching("obstacle"))
        self._partial_wall_state = state
        self._restore_partial_slip()

    def _restore_partial_slip(self):
        """Reapply cached wall data after flux extrapolation, without reinversion."""
        state = self._partial_wall_state
        if state is None:
            return self._apply_partial_slip()
        q = self.solutions[0]
        if "south" in state:
            q[0] = state["south"]
            self._wall_q_north = state["north"]
            self._adv_y[1,0,0] = state["south"]
            self._adv_x[0,0,0] = state["south_tangent"]
            self._adv_y[0,0,-1] = state["north"]
        if "west" in state:
            q[:,0] = state["west"]
            q[0,0] = .5*(state["south"][0]+state["west"][0])
            self._wall_q_east = state["east"]
            self._adv_x[1,0,:,0] = state["west"]
            self._adv_y[0,0,:,0] = state["west_tangent"]
            self._adv_x[0,0,:,-1] = state["east"]
        if self.has_obstacle:
            fx,fy,solid = self._cached_mask_indices()
            self._adv_x[0,0].ravel()[fx] = 0.
            self._adv_y[0,0].ravel()[fy] = 0.
            q.ravel()[solid] = 0.
            if "island_q" in state:
                q.ravel()[state["island_indices"]] = state["island_q"]

    def _laplacian_boundary_conditions(self):
        if self.slip_length is None:
            return super()._laplacian_boundary_conditions()
        def ghost_values():
            mode = next(iter(self.domain.boundary_conditions))
            if mode in {"Periodic", "Channel"}:
                self.periodic_x()
            if mode == "Periodic":
                self.periodic_y()
            else:
                # North/east are omitted boundary nodes, not reflected faces.
                self.u[0,1:-1] = 2*self.u[1,1:-1]-self.u[2,1:-1]
                self.u[-1,1:-1] = self._wall_q_north
            if mode == "Basin":
                self.u[1:-1,0] = 2*self.u[1:-1,1]-self.u[1:-1,2]
                self.u[1:-1,-1] = self._wall_q_east
        self.laplacian_bcs = ghost_values

    def _new_elliptic_solver(
        self,
        boundary_mode: str,
        coefficient: float,
        coordinates: list[float],
    ):
        active = self._active_mask(boundary_mode)
        if (
            coefficient == 0.0
            and boundary_mode == "Periodic"
            and not self.has_obstacle
        ):
            return PoissonSolver(
                shape=(self.domain.ny, self.domain.nx),
                coords=coordinates,
            )
        return _MaskedHelmholtzSolver(
            active,
            self.domain.dx,
            self.domain.dy,
            coefficient,
            periodic_x=boundary_mode in {"Periodic", "Channel"},
            periodic_y=boundary_mode == "Periodic",
        )

    def _build_streamfunction_solver(
        self,
        boundary_mode: str,
        coordinates: list[float],
    ):
        self._relative_vorticity_solver = self._new_elliptic_solver(
            boundary_mode,
            0.0,
            coordinates,
        )
        return self._new_elliptic_solver(
            boundary_mode,
            self.inverse_deformation_radius_squared,
            coordinates,
        )

    def _psi_boundary_conditions(self) -> None:
        """Configure impermeability and the optional Navier wall-PV closure."""
        original_flux_count = len(self.flux_funcs)
        super()._psi_boundary_conditions()
        if next(iter(self.domain.boundary_conditions)) != "Periodic":
            del self.flux_funcs[original_flux_count:]
        if self.slip_length is not None:
            self.flux_funcs.append(self._restore_partial_slip)

    def _initialise_potential_vorticity(self) -> None:
        if not self._needs_pv_initialisation:
            return
        if self._prescribed_pv_initial_condition is not None:
            pv = self._prescribed_pv_initial_condition
            conservative = np.asarray(
                pv(self.X[:-1, :-1], self.Y[:-1, :-1]),
                dtype=np.float64,
            )
            face_x = np.asarray(
                pv(self.X_adv_x, self.Y_adv_x),
                dtype=np.float64,
            )
            face_y = np.asarray(
                pv(self.X_adv_y, self.Y_adv_y),
                dtype=np.float64,
            )
            expected = (self.domain.ny, self.domain.nx)
            for name, values in (
                ("conservative", conservative),
                ("x-face", face_x),
                ("y-face", face_y),
            ):
                if values.shape != expected:
                    raise ValueError(
                        f"The {name} PV initial condition has shape "
                        f"{values.shape}; expected {expected}."
                    )

            self.solutions[0, ...] = conservative
            self._adv_x[:, 0, ...] = face_x
            self._adv_y[:, 0, ...] = face_y
            self._needs_pv_initialisation = False
            return

        relative_vorticity = self.solutions[0].copy()
        self.psi_c[...] = self._relative_vorticity_solver.solve(
            relative_vorticity
        )
        self.solutions[0, ...] = (
            relative_vorticity
            - self.inverse_deformation_radius_squared * self.psi_c
        )
        self._needs_pv_initialisation = False

    def compute_stream_functions(self) -> None:
        """Invert the QG Helmholtz relation and update velocity vertices."""
        self.psi_c[...] = self.poisson_solver.solve(self.potential_vorticity)
        self.psi_v_bcs()
        self.numerical_fluxes()

    def physical_flux(self):
        """Return PV fluxes with zero normal flux at every solid wall."""
        if not hasattr(self, "_flux_work"):
            self._flux_work = (np.empty_like(self._adv_x[0]), np.empty_like(self._adv_y[0]))
        flux_x, flux_y = self._flux_work
        np.multiply(self._adv_x[0], self.adv_vel[0], out=flux_x)
        np.multiply(self._adv_y[0], self.adv_vel[1], out=flux_y)
        if self.has_obstacle:
            fx, fy, _ = self._cached_mask_indices()
            flux_x.ravel()[fx] = 0.
            flux_y.ravel()[fy] = 0.
        boundary_mode = next(iter(self.domain.boundary_conditions))
        if boundary_mode in {"Channel", "Basin"}:
            flux_y[..., 0, :] = 0.0
            flux_y[..., -1, :] = 0.0
        if boundary_mode == "Basin":
            flux_x[..., :, 0] = 0.0
            flux_x[..., :, -1] = 0.0
        return flux_x, flux_y

    def _transport_buffers(self):
        if not hasattr(self, "_transport_work"):
            self._transport_work = tuple(np.empty_like(self._cons) for _ in range(3))
            for name in ("_Q_x", "_Q_y", "_Q"):
                if getattr(self, name, None) is None:
                    setattr(self, name, np.zeros_like(self._cons))
            for name in ("_M_x", "_m_x", "_M_y", "_m_y"):
                if not hasattr(self, name):
                    setattr(self, name, np.empty_like(self._cons))
        return self._transport_work

    def predictor_corrector(self, func):
        """Identical CABARET divergence, using persistent stencil buffers."""
        def update():
            work, _, _ = self._transport_buffers()
            fx, fy = self.physical_flux()
            if _kernels.AVAILABLE:
                _kernels.divergence(fx[0], fy[0], self.domain.dx, self.domain.dy,
                    self.dt, self._cons[0], self._Q_x[0], self._Q_y[0], self._Q[0])
                func()
                return
            _shift_into(fx, self._Q_x, -1, 1)
            self._Q_x -= fx
            self._Q_x /= self.domain.dx
            _shift_into(fy, self._Q_y, -2, 1)
            self._Q_y -= fy
            self._Q_y /= self.domain.dy
            np.add(self._Q_x, self._Q_y, out=self._Q)
            np.multiply(self._Q, .5*self.dt, out=work)
            self._cons += work
            func()
        return update

    def _predictor(self):
        work, increment, _ = self._transport_buffers()
        if _kernels.AVAILABLE:
            _kernels.predictor_bounds(self._adv_x[0, 0], self._adv_y[0, 0], self._cons[0],
                self._Q_x[0], self._Q_y[0], self.dt,
                self._M_x[0], self._m_x[0], self._M_y[0], self._m_y[0])
            return
        for faces, axis, upper, lower, transverse in (
                (self._adv_x, -1, self._M_x, self._m_x, self._Q_y),
                (self._adv_y, -2, self._M_y, self._m_y, self._Q_x)):
            _shift_into(faces[0], work, axis, 1)
            np.maximum(faces[0], self._cons, out=upper)
            np.maximum(upper, work, out=upper)
            np.minimum(faces[0], self._cons, out=lower)
            np.minimum(lower, work, out=lower)
            np.multiply(transverse, self.dt, out=increment)
            upper += increment
            lower += increment

    def second_order_extrapolation(self):
        twice, work, bound = self._transport_buffers()
        if _kernels.AVAILABLE:
            _kernels.extrapolate(self._adv_x[0, 0], self._adv_x[1, 0],
                self._adv_y[0, 0], self._adv_y[1, 0], self._cons[0],
                self._M_x[0], self._m_x[0], self._M_y[0], self._m_y[0], twice[0], work[0])
            return
        np.multiply(self._cons, 2., out=twice)
        eps = np.finfo(np.float64).eps
        for faces, axis, upper, lower in (
                (self._adv_x, -1, self._M_x, self._m_x),
                (self._adv_y, -2, self._M_y, self._m_y)):
            _shift_into(faces[0], work, axis, 1)
            np.subtract(twice, work, out=faces[1])
            np.subtract(twice, faces[0], out=work)
            _shift_into(work, faces[0], axis, -1)
            np.add(lower, eps, out=work)
            np.subtract(upper, eps, out=bound)
            np.clip(faces[1], work, bound, out=faces[1])
            _shift_into(lower, work, axis, -1)
            work += eps
            _shift_into(upper, bound, axis, -1)
            bound -= eps
            np.clip(faces[0], work, bound, out=faces[0])

    def numerical_flux_x(self):
        np.copyto(self._adv_x[0], self._adv_x[1], where=self.adv_vel[0] > 0.)

    def numerical_flux_y(self):
        np.copyto(self._adv_y[0], self._adv_y[1], where=self.adv_vel[1] > 0.)

    def _second_order_deriv(self):
        if not _kernels.AVAILABLE:
            return super()._second_order_deriv()
        if not hasattr(self, "laplacian_u"):
            self.laplacian_u = np.empty(self._cons.shape[-2:])
        _kernels.diffusion(self.u, self.laplacian_u, self.domain.dx**2,
                           self.domain.dy**2, self.dt*self.nu)

    def _average_psi_to_vertices(self):
        mode = next(iter(self.domain.boundary_conditions))
        if _kernels.AVAILABLE:
            _kernels.vertices(self.psi_c, self.psi_v, {"Periodic": 0, "Channel": 1, "Basin": 2}[mode],
                              self._channel_psi_north)
            return
        if not hasattr(self, "_vertex_work"):
            self._vertex_work = tuple(np.empty_like(self.psi_c) for _ in range(3))
        north, east, northeast = self._vertex_work
        _shift_into(self.psi_c, north, -2, -1)
        _shift_into(self.psi_c, east, -1, -1)
        _shift_into(north, northeast, -1, -1)
        mode = next(iter(self.domain.boundary_conditions))
        if mode == "Channel":
            north[-1] = northeast[-1] = self._channel_psi_north
        elif mode == "Basin":
            north[-1] = east[:, -1] = northeast[-1] = northeast[:, -1] = 0.
        np.add(self.psi_c, north, out=self.psi_v)
        self.psi_v += east
        self.psi_v += northeast
        self.psi_v *= .25

    def mask_cmm(self) -> None:
        """Apply Navier wall PV, or retain the legacy manufactured-case mask.

        QG only requires a constant streamfunction on the solid boundary.
        The inherited Navier--Stokes implementation additionally replaces
        boundary vorticity with a no-slip wall closure, which would reject
        valid manufactured solutions having tangential flow along the wall.
        With slip_length=None only solid PV and normal fluxes are removed.
        An explicit slip length additionally supplies the wall vorticity.
        """
        if self.slip_length is not None:
            return self._apply_partial_slip()
        boundary_mode = next(iter(self.domain.boundary_conditions))
        if boundary_mode in {"Channel", "Basin"}:
            self.solutions[0, 0, :] = 0.0
        if boundary_mode == "Basin":
            self.solutions[0, :, 0] = 0.0
        if self.has_obstacle:
            fx, fy, solid = self._cached_mask_indices()
            self._adv_x[0,0].ravel()[fx] = 0.
            self._adv_y[0,0].ravel()[fy] = 0.
            self.solutions[0].ravel()[solid] = 0.

    def update_dynamic_forcing(self) -> None:
        """Apply beta-plane and linear-drag source terms to PV."""
        np.multiply(self.cons_vel[1, 0], self.beta, out=self.forcing)
        np.subtract(self.external_forcing, self.forcing, out=self.forcing)
        if self.linear_drag:
            self.forcing -= self.linear_drag*self.relative_vorticity

    def compute_velocity_field(self):
        """Same CABARET reconstruction, with reusable velocity/source workspaces."""
        u, v = self.adv_vel
        if _kernels.AVAILABLE:
            self._transport_buffers()
            mode = next(iter(self.domain.boundary_conditions))
            bcs = self.domain.boundary_conditions[mode]
            _kernels.velocity(self.psi_v, u[0], v[0], self.adv_vel_raw_prev[0][0],
                self.adv_vel_raw_prev[1][0], self.cons_vel[0, 0], self.cons_vel[1, 0],
                self.domain.dx, self.domain.dy, self.alpha_t,
                {"Periodic": 0, "Channel": 1, "Basin": 2}[mode],
                0. if mode == "Periodic" else bcs[1][2],
                bcs[0][2] if mode == "Basin" else 0.)
            self.update_dynamic_forcing()
            _kernels.limiter_sources(self._adv_x[0, 0], self._adv_y[0, 0],
                self.cons_vel[0, 0], self.cons_vel[1, 0], self._Q[0], self.forcing,
                self.domain.dx, self.domain.dy, self.dt,
                self._M_x[0], self._m_x[0], self._M_y[0], self._m_y[0])
            return
        np.subtract(self.psi_v[:-1],self.psi_v[1:],out=u[0,1:])
        u[0,1:] /= self.domain.dy
        np.subtract(self.psi_v[:,1:],self.psi_v[:,:-1],out=v[0,:,1:])
        v[0,:,1:] /= self.domain.dx
        mode = next(iter(self.domain.boundary_conditions))
        if mode == "Periodic":
            u[0,0] = (self.psi_v[-1]-self.psi_v[0])/self.domain.dy
        else:
            u[0,0] = self.domain.boundary_conditions[mode][1][2]
        if mode == "Basin":
            v[0,:,0] = self.domain.boundary_conditions[mode][0][2]
        else:
            v[0,:,0] = (self.psi_v[:,0]-self.psi_v[:,-1])/self.domain.dx
        if not hasattr(self,"_raw_velocity_work"):
            self._raw_velocity_work = tuple(np.empty_like(a) for a in (u,v))
        for current, previous, raw in zip((u,v),self.adv_vel_raw_prev,self._raw_velocity_work):
            np.copyto(raw,current)
            current *= 1+self.alpha_t
            previous *= self.alpha_t
            current -= previous
            np.copyto(previous,raw)
        np.add(u[...,:-1],u[...,1:],out=self.cons_vel[0,...,1:])
        self.cons_vel[0,...,1:] *= .5
        np.add(v[...,:-1,:],v[...,1:,:],out=self.cons_vel[1,...,1:,:])
        self.cons_vel[1,...,1:,:] *= .5
        self.cons_vel[0,...,0] = .5*(u[...,-1]+u[...,0]) if mode != "Basin" else 0.
        self.cons_vel[1,...,0,:] = .5*(v[...,-1,:]+v[...,0,:]) if mode == "Periodic" else 0.
        self.update_dynamic_forcing()
        tmp_f, tmp_g, increment = self._transport_buffers()
        _shift_into(self._adv_x[0], tmp_f, -1, 1)
        np.subtract(self._adv_x[0], tmp_f, out=tmp_f)
        tmp_f /= self.domain.dx
        _shift_into(self._adv_y[0], tmp_g, -2, 1)
        np.subtract(self._adv_y[0], tmp_g, out=tmp_g)
        tmp_g /= self.domain.dy
        # Each bound pair receives the same increment; calculate it once.
        np.multiply(self.cons_vel[0], tmp_f, out=increment)
        increment += self._Q
        increment += self.forcing
        increment *= self.dt
        self._M_x += increment
        self._m_x += increment
        np.multiply(self.cons_vel[1], tmp_g, out=increment)
        increment += self._Q
        increment += self.forcing
        increment *= self.dt
        self._M_y += increment
        self._m_y += increment

    @property
    def rossby_rate_bound(self):
        """Retain the validated conservative explicit beta source bound."""
        if self._rossby_length is None:
            d = self.domain
            self._rossby_length = max(d.east-d.west,d.north-d.south)
        return abs(self.beta)*self._rossby_length

    def update_dt(self, adv=lambda: None, diff=lambda: None):
        """Include explicit beta/drag stability even for near-rest SI flows."""
        super().update_dt(adv=adv, diff=diff)
        rate = self.rossby_rate_bound + self.linear_drag
        if rate:
            # Raising the advective CFL does not relax the explicit beta
            # source cap beyond the previously exercised value of 0.15.
            self.dt = min(self.dt, min(self.cfl,.15)/rate)
        self.last_step = self.t+self.dt >= self.final_time

    def get_pre_processes(self):
        processes = super().get_pre_processes()
        return (
            processes[:4]
            + [self._initialise_potential_vorticity]
            + processes[4:]
        )

    def terminal_verbose(self, n, v_freq=100):
        if not hasattr(self, "_si_interface"):
            return super().terminal_verbose(n, v_freq)
        if n % v_freq == 0 or self.last_step:
            si = self._si_interface
            print(f"Time-step #{n}: t={si.t:.6g} s, dt={si.dt:.6g} s, "
                  f"max velocity={si.max_velocity_m_s:.6g} m/s", flush=True)


class QuasiGeostrophic(QGSIInterface):
    """Single-layer QG with SI inputs and outputs (times in seconds).

    Use ``units='dimensionless'`` for analytic benchmarks. The numerical
    engine remains an IncNavierStokes subclass and is available through
    ``dimensionless``. Reference scales are 3840 km and 0.01 m/s.
    ``slip_length`` is in metres and applies to all stationary regular and
    grid-aligned obstacle faces. 0 is no slip; np.inf is free slip; None
    preserves the legacy boundary mask for manufactured benchmarks.
    """

    def __init__(self, domain, obstacle=True, obstacle_size=None, *,
                 deformation_radius=None, beta=None, linear_drag=None, nu=None,
                 slip_length=None, units="SI"):
        if units not in {"SI", "dimensionless"}:
            raise ValueError("units must be 'SI' or 'dimensionless'.")
        si = units == "SI"
        s = QG_SCALES
        L, U, time = (s.length, s.velocity, s.time) if si else (1., 1., 1.)
        mesh = scaled_grid(domain, L, U) if si else domain
        radius = (s.length if si else 1.) if deformation_radius is None else deformation_radius
        beta = (DEFAULT_BETA if si else 0.) if beta is None else beta
        drag = (DEFAULT_DRAG if si else 0.) if linear_drag is None else linear_drag
        core = _QuasiGeostrophicCore(mesh, obstacle,
            None if obstacle_size is None else tuple(np.asarray(obstacle_size)/L),
            deformation_radius=radius/L, beta=beta*L*time, linear_drag=drag*time,
            slip_length=None if slip_length is None else slip_length/L)
        if nu is not None or si:
            core.nu = (DEFAULT_NU if nu is None else nu)/(L*U)
        self._attach(core, domain, units)
