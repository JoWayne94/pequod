"""
Incompressible Navier Stokes equations solver class in 2D.

partial omega / partial t + u partial omega / partial x + v partial omega / partial y = (1 / Re) nabla^2 omega + f,
nabla^2 psi = omega.

Todo: 1) Work on elliptic solvers and CMM.
"""

from typing import Callable
from .solver import glob, os, plt, animation
from .adr import Grid, ADR, np, partial
from ..helmsolve.helmsolve import *

from ..CMM.ccm_test import *


class IncNavierStokes(ADR):
    """
    Incompressible Navier Stokes solver on a 2D ``Grid``;
    discretised using the ``CABARET`` and elliptic solver of choice.
    """

    def __init__(
        self,
        domain: Grid,
        obstacle: bool | str = True,
        obstacle_size: tuple[float, float] | None = None,
    ) -> None:
        """
        Main constructor for the ``IncNavierStokes`` class.
        :return: 2D stream function-velocity formulation solver object.
        """

        super().__init__(domain)
        if isinstance(obstacle, str):
            obstacle_kind = obstacle.lower().replace("-", "_")
            aliases = {
                "center": "centered",
                "centre": "centered",
                "southeast": "southeast_bump",
                "se_bump": "southeast_bump",
            }
            obstacle_kind = aliases.get(obstacle_kind, obstacle_kind)
            if obstacle_kind not in {"none", "centered", "southeast_bump"}:
                raise ValueError(
                    "obstacle must be False, True, 'centered', or "
                    "'southeast_bump'."
                )
        else:
            obstacle_kind = "centered" if obstacle else "none"
        self.obstacle_kind = obstacle_kind
        self.has_obstacle = obstacle_kind != "none"
        self._channel_throughflow = 0.0
        self._ic = None
        self.alpha_t = 1.
        self.dt_prev = 0.
        self.adv_vel_raw_prev = tuple(
            component.copy() for component in self.adv_vel
        )

        # Stream function at grid points
        self.psi_c = np.zeros((self.domain.ny, self.domain.nx))
        # Stream function at vertices
        self.psi_v = np.zeros((self.ny_faces, self.nx_faces))
        # Velocity at grid points
        self.cons_vel = np.zeros((2, self.n_vars, self.domain.ny, self.domain.nx))
        # Forcing terms on the RHS defined at grid points
        self.forcing = np.zeros((self.domain.ny, self.domain.nx), dtype=np.float64)

        self.ani_sol = {"omega": self.solutions[0], "psi": self.psi_c}

        # Temporary CMM setup
        self.A = np.array([self.domain.west, self.domain.south])
        self.B = np.array([self.domain.east, self.domain.south])
        self.C = np.array([self.domain.east, self.domain.north])
        self.D = np.array([self.domain.west, self.domain.north])

        nx = int(self.domain.nx)
        ny = int(self.domain.ny)
        domain_width = self.domain.east - self.domain.west
        domain_height = self.domain.north - self.domain.south
        if self.obstacle_kind == "southeast_bump":
            default_size = (0.25 * domain_width, 0.25 * domain_height)
            w, h = default_size if obstacle_size is None else obstacle_size
            x0 = self.domain.east - 0.5 * w
            y0 = self.domain.south + 0.5 * h
        else:
            x0 = 0.5 * (self.domain.west + self.domain.east)
            y0 = 0.5 * (self.domain.south + self.domain.north)
            default_size = (0.0625 * domain_width, 0.125 * domain_height)
            w, h = default_size if obstacle_size is None else obstacle_size
        if not 0.0 < w < domain_width or not 0.0 < h < domain_height:
            raise ValueError(
                "obstacle_size must contain positive physical dimensions "
                "smaller than the domain."
            )
        self.obstacle_center = np.array([x0, y0])
        self.obstacle_width = w
        self.obstacle_height = h

        grid_points = np.stack(
            (self.X[:-1, :-1], self.Y[:-1, :-1]),
            axis=-1,
        )

        if not self.has_obstacle:
            self.obstacle_mask = np.zeros((ny, nx), dtype=bool)
            self.boundary_indices = np.empty((0, 2), dtype=int)
        elif self.obstacle_kind == "southeast_bump":
            self.obstacle_mask, self.boundary_indices = (
                rectangular_corner_bump_mask(
                    self.X[:-1, :-1],
                    self.Y[:-1, :-1],
                    self.domain.west,
                    self.domain.south,
                    self.domain.east,
                    self.domain.north,
                    width=w,
                    height=h,
                )
            )
        else:
            self.obstacle_mask, self.boundary_indices = rectangular_obstacle_mask(
                self.X[:-1, :-1],
                self.Y[:-1, :-1],
                Lx=domain_width,
                Ly=domain_height,
                width=w,
                height=h,
                x0=x0,
                y0=y0,
            )
        if self.has_obstacle and len(self.boundary_indices) == 0:
            raise ValueError(
                "CMM obstacle boundary has no grid-aligned points. "
                "Choose nx and ny so the centered obstacle edges lie on grid points."
            )

        self.obstacle_interior = self.obstacle_mask.copy()
        self.obstacle_interior[
            self.boundary_indices[:, 0],
            self.boundary_indices[:, 1],
        ] = False

        self.points = grid_points[
            self.boundary_indices[:, 0],
            self.boundary_indices[:, 1],
        ]

        # plt.plot(self.points[:, 0], self.points[:, 1], "ro", markersize=5, markerfacecolor="none")
        # plt.xlim(self.domain.west, self.domain.east)
        # plt.ylim(self.domain.south, self.domain.north)
        # plt.show()

        # raise ValueError

        self.Cmat = None
        if self.obstacle_kind == "centered":
            self.Cmat, _ = build_capacitance_matrix_mixed_bc(
                self.A, self.B, self.C, self.D,
                self.points,
                nx,
                ny - 1,
                weights=None,
            )

        self.grid_points2 = np.column_stack([
            self.X[:-1, :-1].ravel(),
            self.Y[:-1, :-1].ravel(),
        ])

        tol = 1.e-11
        x_grid = self.X[:-1, :-1]
        y_grid = self.Y[:-1, :-1]
        x_min = x0 - 0.5 * w
        x_max = x0 + 0.5 * w
        y_min = y0 - 0.5 * h
        y_max = y0 + 0.5 * h

        if self.has_obstacle:
            aligned_edges = (
                (x_min - self.domain.west) / self.domain.dx,
                (x_max - self.domain.west) / self.domain.dx,
                (y_min - self.domain.south) / self.domain.dy,
                (y_max - self.domain.south) / self.domain.dy,
            )
            if not all(
                np.isclose(edge, np.rint(edge), atol=1.0e-12, rtol=0.0)
                for edge in aligned_edges
            ):
                raise ValueError(
                    "The embedded obstacle edges must align with grid points. "
                    "Choose nx and ny compatible with the obstacle dimensions."
                )

        west_boundary = (
            np.isclose(x_grid, x_min, atol=tol)
            & (y_min <= y_grid)
            & (y_grid <= y_max)
        )
        north_boundary = (
            np.isclose(y_grid, y_max, atol=tol)
            & (x_min <= x_grid)
            & (x_grid <= x_max)
        )
        if self.obstacle_kind == "southeast_bump":
            west_boundary &= y_grid > self.domain.south + tol
            east_boundary = np.zeros_like(west_boundary)
            south_boundary = np.zeros_like(west_boundary)
        else:
            east_boundary = (
                np.isclose(x_grid, x_max, atol=tol)
                & (y_min <= y_grid)
                & (y_grid <= y_max)
            )
            south_boundary = (
                np.isclose(y_grid, y_min, atol=tol)
                & (x_min <= x_grid)
                & (x_grid <= x_max)
            )

        if not self.has_obstacle:
            west_boundary.fill(False)
            north_boundary.fill(False)
            east_boundary.fill(False)
            south_boundary.fill(False)
        self.west_bi = np.argwhere(west_boundary)
        self.north_bi = np.argwhere(north_boundary)
        self.east_bi = np.argwhere(east_boundary)
        self.south_bi = np.argwhere(south_boundary)

        # An advective x-face lies between columns i and i + 1.  An
        # advective y-face lies between rows j and j + 1.  Block a face if
        # either adjacent conservative point belongs to the solid.
        self.obstacle_faces_x = np.zeros_like(self.obstacle_mask)
        self.obstacle_faces_y = np.zeros_like(self.obstacle_mask)
        if self.has_obstacle:
            if self.obstacle_kind == "southeast_bump":
                self.obstacle_faces_x[:, :-1] = (
                    self.obstacle_mask[:, :-1]
                    | self.obstacle_mask[:, 1:]
                )
                self.obstacle_faces_x[:, -1] = self.obstacle_mask[:, -1]
            else:
                self.obstacle_faces_x[...] = self.obstacle_mask | np.roll(
                    self.obstacle_mask,
                    -1,
                    axis=-1,
                )
            self.obstacle_faces_y[:-1] = (
                self.obstacle_mask[:-1] | self.obstacle_mask[1:]
            )
            if self.obstacle_kind == "southeast_bump":
                self.obstacle_faces_y[-1] = self.obstacle_mask[-1]

        A0, e1, e2, self.L, self.H = rectangle_frame(self.A, self.B, self.C, self.D)

        x_uv = to_local(self.grid_points2.reshape(-1, 2), A0, e1, e2)
        y_uv = to_local(self.points, A0, e1, e2)

        x_uv[:, 0] = np.mod(x_uv[:, 0], self.L)
        # x_uv[:, 1] = np.mod(x_uv[:, 1], self.H)
        y_uv[:, 0] = np.mod(y_uv[:, 0], self.L)
        # y_uv[:, 1] = np.mod(y_uv[:, 1], self.H)

        # u = x_uv[:, 0]
        # v = x_uv[:, 1]
        # y_u = y_uv[:, 0]
        # y_v = y_uv[:, 1]

        # self.du = u[:, None] - y_u[None, :]
        # self.dv = v[:, None] - y_v[None, :]

        # self.kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.L / nx)
        # self.ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.H / ny)
        # self.grid_boundary_green = self._green_matrix_from_offsets(self.du, self.dv)
        self.grid_boundary_green = None
        if self.obstacle_kind == "centered":
            self.grid_boundary_green = mixed_green_matrix_cross(
                x_uv,
                y_uv,
                self.L,
                self.H,
                nx,
                ny - 1,
            )

    @property
    def re(self):
        """
        :return: Reynolds number.
        """
        return np.inf if self.nu == 0.0 else 1.0 / self.nu

    @re.setter
    def re(self, value: float):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Reynolds number must be finite and positive.")
        self.nu = 1. / value

    @property
    def channel_throughflow(self) -> float:
        """Return the prescribed mean streamwise velocity in a channel."""
        return self._channel_throughflow

    @channel_throughflow.setter
    def channel_throughflow(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("channel_throughflow must be finite.")
        self._channel_throughflow = float(value)

    @property
    def _channel_psi_south(self) -> float:
        return 0.0

    @property
    def _channel_psi_north(self) -> float:
        return -self.channel_throughflow * (
            self.domain.north - self.domain.south
        )

    @property
    def _obstacle_streamfunction(self) -> float:
        if self.obstacle_kind == "southeast_bump":
            return 0.0
        return -self.channel_throughflow * (
            self.obstacle_center[1] - self.domain.south
        )

    def update_alpha(self) -> None:
        """
        Velocity update weighting.
        """
        self.alpha_t = (
            0.0
            if self.dt_prev == 0.0
            else self.dt / (self.dt + self.dt_prev)
        )

    def update_dt_prev(self) -> None:
        self.dt_prev = self.dt

    def update_c(self):
        """
        Compute the unsplit two-dimensional advective rate.

        ``ScalarTransport.update_dt`` expects a wave speed scaled by the
        smallest grid spacing, so this produces
        ``dt = cfl / max(|u| / dx + |v| / dy)``.
        """
        rate = np.max(
            np.abs(self.adv_vel[0]) / self.domain.dx
            + np.abs(self.adv_vel[1]) / self.domain.dy
        )
        self.max_wave_speed = min(self.domain.dx, self.domain.dy) * rate

    def physical_flux(self):
        """
        Return vorticity fluxes with every solid-fluid face closed.
        """
        flux_x, flux_y = super().physical_flux()
        return (
            np.where(self.obstacle_faces_x, 0.0, flux_x),
            np.where(self.obstacle_faces_y, 0.0, flux_y),
        )

    """ Conversion routines. """

    def psi_v_to_cons_vel(self):
        """
        Get velocity field at cell centres.
        :return:
        """
        pass

    def get_pressure(self):
        """
        Get pressure field solution.
        :return:
        """
        pass

    """ Stream function calculations. """

    # def psi_internal(self) -> None:
    #     """
    #     Obtain stream function at vertices by averaging immediate cell centred neighbours.
    #     """
    #     self.psi_v[1:-1, 1:-1] = 0.25 * (self.psi_c[1:, 1:] + self.psi_c[1:, :-1]
    #                                      + self.psi_c[:-1, 1:] + self.psi_c[:-1, :-1])

    # def psi_channel(self) -> None:

    #     self.psi_v[...] = 0.25 * (
    #             self.psi_c
    #             + np.roll(self.psi_c, -1, axis=-1)
    #             + self.psi_c[:, -1]
    #             + np.roll(self.psi_c[:, -1], shift=1)
    #     )
    #     # Stream function at domain corners assumed periodic as all other
    #     # BC types are called after and able to overwrite the periodic BCs
    #     self.psi_v[-1, 0] = self.psi_v[0, 0]
    #     self.psi_v[:, -1] = self.psi_v[:, 0]

    # def psi_basin(self) -> None:

    #     self.psi_v[0, :-1] = 0.25 * (
    #             self.psi_c[0, :]
    #             + np.roll(self.psi_c[0, :], shift=1)
    #             + self.psi_c[-1, :]
    #             + np.roll(self.psi_c[-1, :], shift=1)
    #     )
    #     # Repeated computations for two dofs at the
    #     # bottom left and right domain corners
    #     self.psi_v[0, -1] = self.psi_v[0, 0]
    #     self.psi_v[-1, :] = self.psi_v[0, :]

    def psi_doubly_periodic(self) -> None:
        """
        Stream function computations exclusively for doubly periodic BCs.
        """
        self.psi_c[...] = self.poisson_solver.solve(self.solutions[0])[...]
        self._average_psi_to_vertices()

    def _average_psi_to_vertices(self) -> None:
        """
        Average cell-corner streamfunction values to velocity vertices.

        The x direction is periodic.  A channel's final y vertex uses the
        prescribed north-wall streamfunction instead of wrapping to the
        south wall.
        """
        boundary_mode = next(iter(self.domain.boundary_conditions))
        psi_north = np.roll(self.psi_c, -1, axis=0)
        psi_east = np.roll(self.psi_c, -1, axis=1)
        psi_northeast = np.roll(psi_north, -1, axis=1)
        if boundary_mode == "Channel":
            psi_north[-1, :] = self._channel_psi_north
            psi_northeast[-1, :] = self._channel_psi_north
        elif boundary_mode == "Basin":
            psi_north[-1, :] = 0.0
            psi_east[:, -1] = 0.0
            psi_northeast[-1, :] = 0.0
            psi_northeast[:, -1] = 0.0

        self.psi_v[...] = 0.25 * (
            self.psi_c
            + psi_north
            + psi_east
            + psi_northeast
        )

    def compute_stream_functions(self) -> None:
        """
        Compute stream function values at both grid points and vertices.
        """
        if list(self.domain.boundary_conditions)[0] == 'Periodic':
            self.psi_c[...] = self.poisson_solver.solve(self.solutions[0])[...]
        elif list(self.domain.boundary_conditions)[0] == 'Channel':
            self.psi_c[0, :] = self._channel_psi_south
            self.psi_c[1:, :] = self.poisson_solver.solve(self.solutions[0, 1:, :])[...]
            self.psi_c[...] -= self.channel_throughflow * (
                self.domain.y_coords[:-1, None] - self.domain.south
            )
        elif list(self.domain.boundary_conditions)[0] == 'Basin':
            if self.obstacle_kind == "southeast_bump":
                self.psi_c[...] = self.poisson_solver.solve(
                    self.solutions[0]
                )
            else:
                self.psi_c[...] = 0.0
                self.psi_c[1:, 1:] = self.poisson_solver.solve(
                    self.solutions[0, 1:, 1:]
                )
        else:
            raise ValueError(" BCs not recognised in IncNS object. ")

        if self.obstacle_kind == "centered":
            u_boundary = self.psi_c[
                self.boundary_indices[:, 0],
                self.boundary_indices[:, 1],
            ]
            q = np.linalg.solve(
                self.Cmat,
                self._obstacle_streamfunction - u_boundary,
            )
            self.psi_c[...] += self.capacitance_correction(q).reshape(
                self.domain.ny,
                self.domain.nx,
            )

        self.psi_v_bcs()
        self.numerical_fluxes()

    """ Visualisations. """

    def sol_plt(self, solution):
        """
        Add the omitted boundary points and hide the solid obstacle region.
        """
        obstacle_mask = (
            self.obstacle_mask
            if self.has_obstacle
            else np.zeros_like(self.obstacle_mask)
        )
        if next(iter(self.domain.boundary_conditions)) == "Basin":
            data = np.pad(
                np.asarray(solution),
                ((0, 1), (0, 1)),
                mode="constant",
            )
            mask = np.pad(
                obstacle_mask,
                ((0, 1), (0, 1)),
                mode="constant",
            )
            if self.obstacle_kind == "southeast_bump":
                # The stored grid includes the south wall but omits the east
                # endpoint. Extend the final solid column onto that omitted
                # boundary so the rendered bump is visibly corner-attached.
                mask[:-1, -1] = obstacle_mask[:, -1]
        else:
            data = np.pad(
                np.asarray(solution),
                ((1, 0), (0, 1)),
                mode="wrap",
            )
            mask = np.pad(
                obstacle_mask,
                ((1, 0), (0, 1)),
                mode="wrap",
            )
        return np.ma.array(data, mask=mask)

    def _setup_two_panel_figure(
            self,
            solution,
            z_lims: list[tuple[float, float] | None] = [None, None],
            title: str = "",
            use_tex = True,
    ):
        """
        Create the 1x2 panel figure, add both imshows + colorbars, return the figure and artists.
        :param z_lims: (v_min, v_max) tuples; if None, compute from the arrays.
        :return: fig, {'im1': im1, 'im2': im2, 'title': suptitle_text}
        """
        # constrained_layout avoids manual tight_layout + suptitle clashes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        omega_plot = self.sol_plt(solution["omega"])
        psi_plot = self.sol_plt(solution["psi"])

        if z_lims[0] is None:
            z_lims[0] = (np.ma.min(omega_plot), np.ma.max(omega_plot))
        if z_lims[1] is None:
            z_lims[1] = (np.ma.min(psi_plot), np.ma.max(psi_plot))

        # Left panel
        im1 = ax1.imshow(
            # np.ma.array(solution["omega"], mask=~self.in_mask.T), extent=(self.domain.west, self.domain.east, self.domain.south, self.domain.north),
            omega_plot, extent=(self.domain.west, self.domain.east, self.domain.south, self.domain.north),
            origin="lower", cmap="viridis",
            aspect="auto", vmin=z_lims[0][0], vmax=z_lims[0][1]
        )
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$", rotation=0)
        ax1.set_xlim(self.domain.west, self.domain.east)
        ax1.set_ylim(self.domain.south, self.domain.north)
        ax1.set_aspect("equal", adjustable="box")
        fig.colorbar(im1, ax=ax1, label=r"$\omega$", location="top", fraction=0.046, pad=0.04)

        # Right panel
        im2 = ax2.imshow(
            # np.ma.array(solution["psi"], mask=~self.in_mask.T), extent=(self.domain.west, self.domain.east, self.domain.south, self.domain.north),
            psi_plot, extent=(self.domain.west, self.domain.east, self.domain.south, self.domain.north),
            origin="lower", cmap="viridis",
            aspect="auto", vmin=z_lims[1][0], vmax=z_lims[1][1]
        )
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$", rotation=0)
        ax1.set_xlim(self.domain.west, self.domain.east)
        ax1.set_ylim(self.domain.south, self.domain.north)
        ax2.set_aspect("equal", adjustable="box")
        fig.colorbar(im2, ax=ax2, label=r"$\psi$", location="top", fraction=0.046, pad=0.04)

        title_artist = fig.suptitle(title, fontsize=16, usetex=use_tex)

        try:
            # Fix levels once for consistent animation styling
            n_levels = 10
            psi_levels = np.linspace(z_lims[1][0], z_lims[1][1], n_levels)
            ct2 = ax2.contour(self.domain.x_coords, self.domain.y_coords, psi_plot,
                              levels=psi_levels, linewidths=0.6, colors="k")

            return fig, {
                "im1": im1,
                "im2": im2,
                "ct2": ct2,
                "psi_levels": psi_levels,
                "ax2": ax2,
                "title": title_artist,
            }
        except ValueError:
            return fig, {"im1": im1, "im2": im2, "title": title_artist}

    def save_animation(
        self, ani_name: str = "solutions.gif", del_data: bool = True) -> None:

        omega_list = sorted(glob.glob(os.path.join(self.data_dir, "omega_*.dat")))
        psi_list = sorted(glob.glob(os.path.join(self.data_dir, "psi_*.dat")))
        n_frames = len(omega_list)
        assert n_frames == len(psi_list), " Number of vorticity and stream function data files mismatch. "

        # Precompute global z-limits
        _, sample_o = self.load_data(omega_list[0])
        _, sample_p = self.load_data(psi_list[0])

        # Fix z_lims based on first frame
        # o_lims = (sample_o.min(), sample_o.max())
        # p_lims = (sample_p.min(), sample_p.max())

        # Compute global lims across all frames
        o_mins, o_maxs, p_mins, p_maxs = [], [], [], []
        for fo, fp in zip(omega_list, psi_list):
            _, o = self.load_data(fo)
            _, p = self.load_data(fp)
            o_plot = self.sol_plt(o)
            p_plot = self.sol_plt(p)
            o_mins.append(np.ma.min(o_plot)); o_maxs.append(np.ma.max(o_plot))
            p_mins.append(np.ma.min(p_plot)); p_maxs.append(np.ma.max(p_plot))
        o_lims = (min(o_mins), max(o_maxs))
        p_lims = (min(p_mins), max(p_maxs))

        # Build the figure once
        title = f"{self.domain.nx}$\\times${self.domain.ny} Grid. Time = {0.0:.2f} s"
        fig, artists = self._setup_two_panel_figure({"omega": sample_o, "psi": sample_p},
                                                    [o_lims, p_lims], title, use_tex=True)

        # If blit=True, include the title in returned artists
        def update(frame_idx):
            filename_o = omega_list[frame_idx]
            filename_p = psi_list[frame_idx]
            t_val, omega = self.load_data(filename_o)
            _, psi = self.load_data(filename_p)

            # Update image data
            # artists["im1"].set_data(np.ma.array(omega, mask=~self.in_mask.T))
            # artists["im2"].set_data(np.ma.array(psi, mask=~self.in_mask.T))
            omega_plot = self.sol_plt(omega)
            psi_plot = self.sol_plt(psi)
            artists["im1"].set_data(omega_plot)
            artists["im2"].set_data(psi_plot)

            artists["im1"].set_clim(np.ma.min(omega_plot), np.ma.max(omega_plot))
            artists["im2"].set_clim(np.ma.min(psi_plot), np.ma.max(psi_plot))

            # Re-draw psi contours with the same fixed levels
            if "ct2" in artists.keys():
                qcs = artists["ct2"]
                try:
                    qcs.remove()  # removes the whole contour set in one go
                except ValueError:
                    # Fallback for older Matplotlib where .remove() didn’t exist
                    cols = getattr(qcs, "collections", None)
                    if cols is not None:
                        for c in cols:
                            c.remove()

                artists["ct2"] = artists["ax2"].contour(
                    self.domain.x_coords, self.domain.y_coords, psi_plot,
                    levels=artists["psi_levels"], linewidths=0.6, colors="k"
                )

            artists["title"].set_text(f"{self.domain.nx}$\\times${self.domain.ny} Grid. Time = {t_val:.2f} s")
            return artists["im1"], artists["im2"], artists["title"]

        ani = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=80, blit=False  # blit=False for suptitle updates
        )

        if ani_name[-3:] == "gif":
            ani.save(os.path.join(self.caller_dir, ani_name), writer="pillow")
        elif ani_name[-3:] == "mp4":
            ani.save(os.path.join(self.caller_dir, ani_name), writer="ffmpeg", fps=15)
        else:
            raise NotImplementedError(" Unsupported animation format. ")
        plt.close(fig)

        # Clean up the data files after animation creation
        if del_data:
            for f_name_o, f_name_p in zip(omega_list, psi_list):
                os.remove(f_name_o)
                os.remove(f_name_p)

    """ Initial conditions. """

    def set_ivp(self, ic: Callable = None, var: int = 0, visual: int = 0) -> None:
        """
        Setter for the Initial Value Problem (IVP).
        :param ic:  Initial conditions for the velocity field.
        :param var: Variable index.
        :param visual: Plot switch.
        :return: None
        """

        # Initial datum/conditions defined
        if ic is not None:
            self._ic = ic
            self.cons_vel[0][var], self.cons_vel[1][var] = ic(self.X[:-1, :-1], self.Y[:-1, :-1])

            self.adv_vel = ic
            _, v = ic(self.X_adv_x, self.Y_adv_x)
            u, _ = ic(self.X_adv_y, self.Y_adv_y)

            dvdx = (v - np.roll(v, 1, axis=-1)) / self.domain.dx
            dudy = (u - np.roll(u, 1, axis=-2)) / self.domain.dy
            self.solutions[var] = dvdx - dudy
            self.adv_vel_raw_prev = tuple(
                component.copy() for component in self.adv_vel
            )

            # Garbage collection
            # u.resize(0), v.resize(0)

        if visual == 1:
            raise ValueError(" 1D simulations not supported. ")
        elif visual == 2:
            self.visualise_sol(r"Initial datum")
        else:
            pass

    """ End of ICs and start of boundary conditions. """

    def _validate_streamfunction_configuration(
        self,
        boundary_mode: str,
    ) -> None:
        """Validate the Poisson formulation selected by this solver."""
        if self.obstacle_kind == "centered" and boundary_mode != "Channel":
            raise NotImplementedError(
                "The centered embedded obstacle uses the channel CMM and "
                "therefore requires Channel boundary conditions."
            )
        if (
            self.obstacle_kind == "southeast_bump"
            and boundary_mode != "Basin"
        ):
            raise NotImplementedError(
                "The southeast attached bump requires Basin boundary "
                "conditions."
            )
        if self.channel_throughflow != 0.0 and boundary_mode != "Channel":
            raise NotImplementedError(
                "channel_throughflow is only available with Channel "
                "boundary conditions."
            )

    def _build_streamfunction_solver(
        self,
        boundary_mode: str,
        coordinates: list[float],
    ):
        """Build the Poisson inverse used by the vorticity formulation."""
        if boundary_mode == "Periodic":
            return PoissonSolver(
                shape=(self.domain.ny, self.domain.nx),
                coords=coordinates,
            )
        if boundary_mode == "Channel":
            return PoissonSolverChannel(
                shape=(self.domain.ny - 1, self.domain.nx),
                coords=coordinates,
            )
        if boundary_mode == "Basin":
            if self.obstacle_kind == "southeast_bump":
                return PoissonSolverBasinMasked(
                    ~self.obstacle_mask,
                    coordinates,
                )
            return PoissonSolverBasin(
                shape=(self.domain.ny - 1, self.domain.nx - 1),
                coords=coordinates,
            )
        raise ValueError(f"Unsupported boundary mode: {boundary_mode}.")

    def _psi_boundary_conditions(self) -> None:
        """
        Configure the elliptic solve and wall-vorticity closures.
        """
        boundary_mode = next(iter(self.domain.boundary_conditions))
        self._validate_streamfunction_configuration(boundary_mode)

        coordinates = [
            self.domain.west,
            self.domain.south,
            self.domain.east,
            self.domain.north,
        ]
        self.poisson_solver = self._build_streamfunction_solver(
            boundary_mode,
            coordinates,
        )

        X_v, Y_v = np.meshgrid(self.x_coords_faces, self.y_coords_faces)
        u_v, v_v = self._ic(X_v, Y_v)
        dvdx_v = (v_v - np.roll(v_v, 1, axis=-1)) / self.domain.dx
        dudy_v = (u_v - np.roll(u_v, 1, axis=-2)) / self.domain.dy

        psi_funcs = [self._average_psi_to_vertices]

        self._adv_x[0][0] = ((self.cons_vel[1][0] - np.roll(self.cons_vel[1][0], shift=1, axis=1)) / self.domain.dx) - dudy_v
        self._adv_y[0][0] = dvdx_v - ((self.cons_vel[0][0] - np.roll(self.cons_vel[0][0], shift=1, axis=0)) / self.domain.dy)

        for bc, bv in self.domain.boundary_conditions.items():
            if bc in {"Channel", "Basin"}:
                self._adv_y[0][0][:-1, :] = (dvdx_v[:-1, :] -
                                            ((self.cons_vel[0][0][1:, :] - self.cons_vel[0][0][:-1, :]) / self.domain.dy))

                for i in (1, 3):
                    bi = (i == 1) - 1  # 0, -1

                    a = bv[i][0]
                    b = bv[i][1]
                    c = bv[i][2]
                    d = a * b + 0.5 * self.domain.dy

                    if i == 1:
                        # South
                        def slip_s(tmp_bi=bi, tmp_c=c, tmp_d=d):

                            self._cons[0, tmp_bi, :] = (
                                (
                                    self.psi_c[tmp_bi + 1, :]
                                    - self._channel_psi_south
                                )
                                / (self.domain.dy * tmp_d)
                                + (1, -1)[tmp_bi] * tmp_c / tmp_d
                            )
                            self._adv_y[1, ..., tmp_bi, :] = self._cons[0, tmp_bi, :]
                            self._adv_x[0, ..., tmp_bi, :] = 0.5 * (self._cons[0, tmp_bi, :]
                                                                    + np.roll(self._cons[0, tmp_bi, :], -1, axis=-1))
                        self.flux_funcs.append(slip_s)

                    else:
                        # North
                        self._adv_y[0][0][-1, :] = (dvdx_v[-1, :] -
                                                ((c * np.ones(self.domain.nx) - self.cons_vel[0][0][-1, :]) / self.domain.dy))

                        def slip_n(tmp_bi=bi, tmp_c=c, tmp_d=d):

                            self._adv_y[0, ..., tmp_bi, :] = (
                                (
                                    self.psi_c[tmp_bi, :]
                                    - self._channel_psi_north
                                )
                                / (self.domain.dy * tmp_d)
                                + (1, -1)[tmp_bi] * tmp_c / tmp_d
                            )
                        self.flux_funcs.append(slip_n)

            if bc == "Basin":
                self._adv_x[0][0][:, 1:] = (
                    (
                        self.cons_vel[1][0][:, 1:]
                        - self.cons_vel[1][0][:, :-1]
                    )
                    / self.domain.dx
                    - dudy_v[:, 1:]
                )

                for i in (0, 2):
                    bi = -(i != 0)

                    a = bv[i][0]
                    b = bv[i][1]
                    c = bv[i][2]
                    d = a * b + 0.5 * self.domain.dx

                    if i == 0:
                        # West
                        def slip_w(tmp_bi=bi, tmp_c=c, tmp_d=d):

                            self._cons[0, :, tmp_bi] = (
                                self.psi_c[:, tmp_bi + 1]
                                / (self.domain.dx * tmp_d)
                                - tmp_c / tmp_d
                            )
                            self._adv_x[1, ..., tmp_bi] = self._cons[0, :, tmp_bi]
                            self._adv_y[0, ..., tmp_bi] = 0.5 * (self._cons[0, :, tmp_bi]
                                                                + np.roll(self._cons[0, :, tmp_bi], -1, axis=-1))
                        self.flux_funcs.append(slip_w)

                    else:
                        # East
                        def slip_e(tmp_bi=bi, tmp_c=c, tmp_d=d):

                            self._adv_x[0, ..., tmp_bi] = (
                                self.psi_c[:, tmp_bi]
                                / (self.domain.dx * tmp_d)
                                + tmp_c / tmp_d
                            )
                        self.flux_funcs.append(slip_e)

        self.psi_v_bcs = lambda: [psi_bc() for psi_bc in psi_funcs]
        self.numerical_fluxes()

        # Garbage collection
        # X_v.resize(0), Y_v.resize(0), u_v.resize(0), v_v.resize(0)

    """ End of boundary conditions. """

    def set_forcing_terms(self, f: Callable = None) -> None:

        if f is not None:
            for j in np.arange(self.domain.ny):
                for i in np.arange(self.domain.nx):
                    self.forcing[j][i] = f(
                        self.domain.x_coords[i], self.domain.y_coords[j]
                    )

    def compute_velocity_field(self) -> None:

        # Convert psi at vertices to velocity field at half time-steps
        self.adv_vel[0][0][1:, :] = (self.psi_v[:-1, :] - self.psi_v[1:, :]) / self.domain.dy
        self.adv_vel[1][0][:, 1:] = (self.psi_v[..., 1:] - self.psi_v[..., :-1]) / self.domain.dx

        for bc, bv in self.domain.boundary_conditions.items():
            if bc == 'Periodic':
                self.adv_vel[0][0][0, :] = (self.psi_v[-1, :] - self.psi_v[0, :]) / self.domain.dy
                self.adv_vel[1][0][:, 0] = (self.psi_v[..., 0] - self.psi_v[..., -1]) / self.domain.dx
            elif bc == 'Channel':
                self.adv_vel[0][0][0, :] = bv[1][2] * np.ones(self.domain.nx)
                self.adv_vel[1][0][:, 0] = (self.psi_v[..., 0] - self.psi_v[..., -1]) / self.domain.dx
            else:
                self.adv_vel[0][0][0, :] = bv[1][2] * np.ones(self.domain.nx)
                self.adv_vel[1][0][:, 0] = bv[0][2] * np.ones(self.domain.ny)

        raw_velocity = tuple(
            component.copy() for component in self.adv_vel
        )

        # Second-order velocity extrapolation from raw streamfunction
        # velocities.  Keep the raw history separate from the extrapolated
        # field to avoid recursively extrapolating an extrapolation.
        self.adv_vel[0][...] = (
            (1.0 + self.alpha_t) * raw_velocity[0]
            - self.alpha_t * self.adv_vel_raw_prev[0]
        )
        self.adv_vel[1][...] = (
            (1.0 + self.alpha_t) * raw_velocity[1]
            - self.alpha_t * self.adv_vel_raw_prev[1]
        )
        self.adv_vel_raw_prev = raw_velocity

        # Recover velocity field at grid points
        self.cons_vel[0][..., 1:] = 0.5 * (self.adv_vel[0][..., :-1] + self.adv_vel[0][..., 1:])
        self.cons_vel[1][..., 1:, :] = 0.5 * (self.adv_vel[1][..., :-1, :] + self.adv_vel[1][..., 1:, :])

        # Doubly periodic BCs for now
        self.cons_vel[0][..., 0] = 0.5 * (self.adv_vel[0][..., -1] + self.adv_vel[0][..., 0])
        self.cons_vel[1][..., 0, :] = 0.5 * (self.adv_vel[1][..., -1, :] + self.adv_vel[1][..., 0, :])
        if list(self.domain.boundary_conditions)[0] != 'Periodic':
            self.cons_vel[1][..., 0, :] = np.zeros(self.domain.nx)
        if list(self.domain.boundary_conditions)[0] == 'Basin':
            self.cons_vel[0][..., 0] = np.zeros(self.domain.ny)

        self.update_dynamic_forcing()

        # Add source terms to maximum principles
        tmp_f = (self._adv_x[0] - np.roll(self._adv_x[0], 1, axis=-1)) / self.domain.dx
        tmp_g = (self._adv_y[0] - np.roll(self._adv_y[0], 1, axis=-2)) / self.domain.dy

        self._M_x += self.dt * (self._Q + self.cons_vel[0] * tmp_f + self.forcing)
        self._m_x += self.dt * (self._Q + self.cons_vel[0] * tmp_f + self.forcing)
        self._M_y += self.dt * (self._Q + self.cons_vel[1] * tmp_g + self.forcing)
        self._m_y += self.dt * (self._Q + self.cons_vel[1] * tmp_g + self.forcing)

    def update_dynamic_forcing(self) -> None:
        """Hook for formulations with state-dependent source terms."""
        return None

    def mask_cmm(self):

        if not self.has_obstacle:
            return

        # self.solutions[0][~self.in_mask.T] = 0.
        self._adv_x[0][0][self.obstacle_faces_x] = 0.
        self._adv_y[0][0][self.obstacle_faces_y] = 0.

        self.solutions[0][self.obstacle_interior] = 0.0

        wall_sum = np.zeros_like(self.solutions[0])
        wall_count = np.zeros_like(self.solutions[0])

        def add_wall_values(indices, values):
            if len(indices) == 0:
                return
            rows = indices[:, 0]
            cols = indices[:, 1]
            np.add.at(wall_sum, (rows, cols), values)
            np.add.at(wall_count, (rows, cols), 1.0)

        add_wall_values(
            self.west_bi,
            (
                (2.0 / self.domain.dx**2)
                * (
                    self.psi_c[self.west_bi[:, 0], self.west_bi[:, 1] - 1]
                    - self._obstacle_streamfunction
                )
            ),
        )
        add_wall_values(
            self.east_bi,
            (
                (2.0 / self.domain.dx**2)
                * (
                    self.psi_c[self.east_bi[:, 0], self.east_bi[:, 1] + 1]
                    - self._obstacle_streamfunction
                )
            ),
        )
        add_wall_values(
            self.south_bi,
            (
                (2.0 / self.domain.dy**2)
                * (
                    self.psi_c[self.south_bi[:, 0] - 1, self.south_bi[:, 1]]
                    - self._obstacle_streamfunction
                )
            ),
        )
        add_wall_values(
            self.north_bi,
            (
                (2.0 / self.domain.dy**2)
                * (
                    self.psi_c[self.north_bi[:, 0] + 1, self.north_bi[:, 1]]
                    - self._obstacle_streamfunction
                )
            ),
        )

        wall_mask = wall_count > 0.0
        self.solutions[0][wall_mask] = wall_sum[wall_mask] / wall_count[wall_mask]

        # self.solutions[0][self.rows, self.cols] += (4. / self.domain.dx * self.domain.dx) * self.psi_c[
        #     self.rows, self.cols - 1] - (4. / self.domain.dy * self.domain.dy) * self.psi_c[
        #     self.rows + 1, self.cols]

        # self.psi_v[self.rows, self.cols] = 0.

        # plt.plot(self.X[r0:r1 + 1, c0], self.Y[r0:r1 + 1, c0], "ro")
        # plt.plot(self.X[r1, c0:c1 + 1], self.Y[r1, c0:c1 + 1], "go")
        # plt.plot(self.X[r1, c0:c1 + 1], self.Y[r0:r1 + 1, c0], "bo")

    def _green_matrix_from_offsets(self, du, dv):

        green = np.zeros_like(du, dtype=float)

        for kyi in self.ky:
            for kxi in self.kx:
                k2 = kxi * kxi + kyi * kyi
                if k2 == 0.0:
                    continue

                phase = kxi * du + kyi * dv
                green += np.cos(phase) / k2

        green /= (self.L * self.H)
        return green

    def capacitance_correction(self, q):

        return self.grid_boundary_green @ q

    def _corrector(self):
        self.solutions[0] += self.laplacian_u
        self.solutions[0] += self.dt * self.forcing

    def get_pre_processes(self):
        return [
            partial(self.update_dt, adv=self.update_c, diff=self.diffusion_dt),
            self._boundary_conditions,
            self._laplacian_boundary_conditions,
            self._psi_boundary_conditions,
            self.compute_stream_functions,
            self.mask_cmm,
        ]

    def get_solve_sequence(self):
        return [
            partial(self.update_dt, adv=self.update_c, diff=self.diffusion_dt),
            self.update_alpha,
            self.predictor,
            self.numerical_fluxes,
            self.compute_stream_functions,
            self.mask_cmm,
            self.compute_velocity_field,
            # self.mask_cmm,
            self.update_maximum_principle,
            self.second_order_extrapolation,
            self.numerical_fluxes,
            self.corrector,
            self.compute_stream_functions,
            self.mask_cmm,
            self.update_dt_prev,
        ]
