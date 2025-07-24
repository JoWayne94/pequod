"""
The CABARET on a 2D ``Grid``.

Todo: 1) Write getters for solutions
"""

from typing import Any, Self, Sequence

from .grid import Grid, np, uint8
from .pq_types import eps64


class Gulf(Grid):
    """
    Second-order generalised upwind leapfrog (GULF) on a 2D uniform ``Grid``.
    """

    def __init__(self, nx, ny, west, south, east, north, index) -> None:
        """
        Main constructor for GULF.
        :return: 2D CABARET derived from ``Grid`` with the given shape.
        """

        super().__init__(nx, ny, west, south, east, north, index)
        self.m_nvars = 1
        self.m_dt = 1.0
        self.m_cfl = 0.5

        self.X, self.Y = np.meshgrid(self.get_x_coords, self.get_y_coords)
        self.X_adv_x, self.Y_adv_x = np.meshgrid(
            self.get_x_coords_faces, self.get_y_coords
        )
        self.X_adv_y, self.Y_adv_y = np.meshgrid(
            self.get_x_coords, self.get_y_coords_faces
        )

        self.predictor = self.predictor_corrector(self.predictor)
        self.corrector = self.predictor_corrector(self.corrector)

        # Boundary value problem (BVP) - default settings are doubly periodic
        self.m_boundary_conditions = ["Periodic", "Periodic", "Periodic", "Periodic"]
        self.m_boundary_values = [None, None, None, None]

        # Conservative-type data
        self.m_cons = np.zeros((self.m_nvars, self.m_ny, self.m_nx))

        # Advection-type data
        self.m_adv_x = np.zeros((2, self.m_nvars, self.m_ny, self.m_nx_adv))
        self.m_adv_y = np.zeros((2, self.m_nvars, self.m_ny_adv, self.m_nx))
        self.m_adv_vel = None

        # Nonlinear flux correction variables
        self.numerical_fluxes = None
        self._M_x, self._m_x, self._M_y, self._m_y = None, None, None, None
        self._Q_x = np.empty_like(self.m_cons)
        self._Q_y = np.empty_like(self.m_cons)

    @property
    def nvars(self):
        """
        Getter for the number of variables in equations.
        :return: Number of variables.
        """
        return self.m_nvars

    @nvars.setter
    def nvars(self, value: int):
        """
        Setter for the number of variables in equations.
        """
        self.m_nvars = uint8(value)

    @property
    def dt(self):
        """
        Getter for the time-step size.
        :return: Time-step size.
        """
        return self.m_dt

    @dt.setter
    def dt(self, value: float):
        """
        Setter for the time-step size.
        """
        self.m_dt = value

    @property
    def cfl(self):
        """
        Getter for the Courant-Friedrichs-Lewy number.
        :return: CFL number.
        """
        return self.m_cfl

    @cfl.setter
    def cfl(self, value: float):
        """
        Setter for the CFL number.
        """
        self.m_cfl = value

    @property
    def solutions(self):
        """
        Getter for solution values.
        :return: Conservative-type solutions.
        """
        return self.m_cons

    """ BVP methods. """

    def bcs(
        self, west: str = None, south: str = None, east: str = None, north: str = None
    ) -> Self:
        tmp_list = [west, south, east, north]

        new_bcs = [
            tmp_list[i] if tmp_list[i] is not None else self.m_boundary_conditions[i]
            for i in range(4)
        ]

        return self.set_bvp(new_bcs, self.m_boundary_values)

    def bvs(
        self,
        west: float = None,
        south: float = None,
        east: float = None,
        north: float = None,
    ) -> Self:
        tmp_list = [west, south, east, north]

        new_bvs = [
            tmp_list[i] if tmp_list[i] is not None else self.m_boundary_values[i]
            for i in range(4)
        ]

        return self.set_bvp(self.m_boundary_conditions, new_bvs)

    def set_bvp(self, bcs: list[str], bvs: list[float | None]) -> None:
        assert len(bvs) == len(bcs) == 4

        tmp = [2, 3, 0, 1]
        b_names = ["Western", "Southern", "Eastern", "Northern"]

        for i, bc in enumerate(bcs):
            if bc not in ["Dirichlet", "Neumann", "Periodic"]:
                raise NotImplementedError(f" Invalid boundary condition {bc}. ")

            if bc == "Periodic" and bcs[tmp[i]] != "Periodic":
                raise ValueError(
                    f" If the {b_names[i]} boundary is Periodic, then the {b_names[tmp[i]]} boundary must also be Periodic. \n"
                    f" Please change either conditions. "
                )

            if (
                bc != "Periodic"
                and bvs[i] is None
                and self.m_boundary_values[i] is not None
            ):
                raise UserWarning(
                    f" The assigned {b_names[i]} boundary condition is {bc} but data given is {None}. \n"
                    f" {b_names[i]} boundary data will fall back to {self.m_boundary_values[i]}. "
                )
            else:
                self.m_boundary_values[i] = bvs[i]

            self.m_boundary_conditions[i] = bc

            if bc == "Periodic" and bvs[i] is not None:
                raise UserWarning(
                    f" The {b_names[i]} boundary is Periodic. Boundary data assigned will be overwritten. "
                )

    def enforce_boundary_conditions(
        self,
        periodic_x: Sequence[Any] = None,
        periodic_y: Sequence[Any] = None,
        internal_x: Sequence[Any] = None,
        internal_y: Sequence[Any] = None,
        dirichlet: tuple[
            Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]
        ] = None,
        neumann: tuple[
            Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]
        ] = None,
    ) -> None:
        """
        Enforce boundary conditions in the following order: West -> East -> South -> North.
        :return: None
        """

        flux_funcs = []

        if self.m_boundary_conditions[0] == self.m_boundary_conditions[2] == "Periodic":
            if periodic_x is not None:
                flux_funcs.append(*periodic_x)
            flux_funcs.append(self.numerical_flux_periodic_x)

        else:
            if internal_x is not None:
                flux_funcs.append(*internal_x)

            for i in (0, 2):
                bi = -(i != 0)

                if self.m_boundary_conditions[i] == "Dirichlet":
                    if dirichlet is not None:
                        flux_funcs.append(*dirichlet[i])

                    def dirichlet_x(tmp_i=i, tmp_bi=bi):
                        self.m_adv_x[0, ..., tmp_bi] = self.m_boundary_values[
                            tmp_i
                        ] * np.ones(self.m_ny)

                    flux_funcs.append(dirichlet_x)

                else:
                    if neumann is not None:
                        flux_funcs.append(*neumann[i])

                    if i == 0:

                        def neumann_x():
                            mask = self.m_adv_vel[0][..., 0] > 0.0
                            self.m_adv_x[0, ..., 0][mask] = self.m_adv_x[1, ..., 1][
                                mask
                            ]

                    else:

                        def neumann_x():
                            self.m_adv_x[0, ..., -1] = self.m_adv_x[1, ..., -1].copy()
                            mask = self.m_adv_vel[0][..., -1] < 0.0
                            self.m_adv_x[0, ..., -1][mask] = self.m_adv_x[0, ..., -2][
                                mask
                            ]

                    flux_funcs.append(neumann_x)

            flux_funcs.append(self.numerical_flux_x)

        if self.m_boundary_conditions[1] == self.m_boundary_conditions[3] == "Periodic":
            if periodic_y is not None:
                flux_funcs.append(*periodic_y)
            flux_funcs.append(self.numerical_flux_periodic_y)

        else:
            if internal_y is not None:
                flux_funcs.append(*internal_y)

            for i in (1, 3):
                bi = (i == 1) - 1

                if self.m_boundary_conditions[i] == "Dirichlet":
                    if dirichlet is not None:
                        flux_funcs.append(*dirichlet[i])

                    def dirichlet_y(tmp_i=i, tmp_bi=bi):
                        self.m_adv_y[0, ..., tmp_bi, :] = self.m_boundary_values[
                            tmp_i
                        ] * np.ones(self.m_nx)

                    flux_funcs.append(dirichlet_y)

                else:
                    if neumann is not None:
                        flux_funcs.append(*neumann[i])

                    if i == 1:

                        def neumann_y():
                            mask = self.m_adv_vel[1][..., 0, :] > 0.0
                            self.m_adv_y[0, ..., 0, :][mask] = self.m_adv_y[
                                1, ..., 1, :
                            ][mask]

                    else:

                        def neumann_y():
                            self.m_adv_y[0, ..., -1, :] = self.m_adv_y[
                                1, ..., -1, :
                            ].copy()
                            mask = self.m_adv_vel[1][..., -1, :] < 0.0
                            self.m_adv_y[0, ..., -1, :][mask] = self.m_adv_y[
                                0, ..., -2, :
                            ][mask]

                    flux_funcs.append(neumann_y)

            flux_funcs.append(self.numerical_flux_y)

        self.numerical_fluxes = lambda: [update_flux() for update_flux in flux_funcs]

    """ End of BVP methods and start of CABARET routines. """

    def physical_flux(self):
        """
        Template of physical fluxes for subclass solvers.
        :return: Convert advection-type data to flux values.
        """
        return self.m_adv_x[0], self.m_adv_y[0]

    def predictor_corrector(self, func):
        def wrapper() -> None:
            f_x, f_y = self.physical_flux()

            self._Q_x = (-f_x[..., 1:] + f_x[..., :-1]) / self.m_dx
            self._Q_y = (-f_y[..., 1:, :] + f_y[..., :-1, :]) / self.m_dy

            self.m_cons += 0.5 * self.dt * (self._Q_x + self._Q_y)  # .copy()

            func()

        return wrapper

    def predictor(self):
        """
        Compute M and m values required for nonlinear flux correction after the predictor step.
        :return: None
        """
        self._M_x = np.maximum.reduce(
            [self.m_adv_x[0, ..., :-1], self.m_cons, self.m_adv_x[0, ..., 1:]]
        )
        self._m_x = np.minimum.reduce(
            [self.m_adv_x[0, ..., :-1], self.m_cons, self.m_adv_x[0, ..., 1:]]
        )
        self._M_y = np.maximum.reduce(
            [self.m_adv_y[0, ..., :-1, :], self.m_cons, self.m_adv_y[0, ..., 1:, :]]
        )
        self._m_y = np.minimum.reduce(
            [self.m_adv_y[0, ..., :-1, :], self.m_cons, self.m_adv_y[0, ..., 1:, :]]
        )

        self._M_x += self.dt * self._Q_y
        self._m_x += self.dt * self._Q_y
        self._M_y += self.dt * self._Q_x
        self._m_y += self.dt * self._Q_x

    def corrector(self):
        pass

    def second_order_extrapolation(self) -> None:
        # Extrapolate from the west/south
        self.m_adv_x[1, ..., 1:] = 2.0 * self.m_cons - self.m_adv_x[0, ..., :-1]
        self.m_adv_y[1, ..., 1:, :] = 2.0 * self.m_cons - self.m_adv_y[0, ..., :-1, :]

        # Extrapolate from the east/north
        self.m_adv_x[0, ..., :-1] = 2.0 * self.m_cons - self.m_adv_x[0, ..., 1:].copy()
        self.m_adv_y[0, ..., :-1, :] = (
            2.0 * self.m_cons - self.m_adv_y[0, ..., 1:, :].copy()
        )

        # Upper and lower bounds from the maximum principle
        self.m_adv_x[1, ..., 1:] = np.clip(
            self.m_adv_x[1, ..., 1:], self._m_x + eps64, self._M_x - eps64
        )
        self.m_adv_x[0, ..., :-1] = np.clip(
            self.m_adv_x[0, ..., :-1], self._m_x + eps64, self._M_x - eps64
        )

        self.m_adv_y[1, ..., 1:, :] = np.clip(
            self.m_adv_y[1, ..., 1:, :], self._m_y + eps64, self._M_y - eps64
        )
        self.m_adv_y[0, ..., :-1, :] = np.clip(
            self.m_adv_y[0, ..., :-1, :], self._m_y + eps64, self._M_y - eps64
        )

    """ Numerical flux methods. """

    def numerical_flux_x(self) -> None:
        """
        Upwind flux calculations for internal cell faces in the x-direction.
        :return: None
        """
        # Update in x-direction
        mask = self.m_adv_vel[0][..., 1:-1] > 0.0
        self.m_adv_x[0, ..., 1:-1][mask] = self.m_adv_x[1, ..., 1:-1][mask]  # .copy()

    def numerical_flux_y(self) -> None:
        """
        Upwind flux calculations for internal cell faces in the y-direction.
        :return: None
        """
        # Update in y-direction
        mask = self.m_adv_vel[1][..., 1:-1, :] > 0.0
        self.m_adv_y[0, ..., 1:-1, :][mask] = self.m_adv_y[1, ..., 1:-1, :][mask]

    def numerical_flux_periodic_x(self) -> None:
        """
        Upwind flux calculations in the x-direction with periodic boundary conditions.
        :return: None
        """
        # Update in x-direction
        mask = self.m_adv_vel[0][..., :-1] > 0.0
        self.m_adv_x[0, ..., :-1][mask] = np.roll(self.m_adv_x[1, ..., 1:], 1, axis=-1)[
            mask
        ]  # .copy()

        # Periodic BC
        self.m_adv_x[0, ..., -1] = self.m_adv_x[0, ..., 0].copy()

    def numerical_flux_periodic_y(self) -> None:
        """
        Upwind flux calculations in the y-direction with periodic boundary conditions.
        :return: None
        """
        # Update in y-direction
        mask = self.m_adv_vel[1][..., :-1, :] > 0.0
        self.m_adv_y[0, ..., :-1, :][mask] = np.roll(
            self.m_adv_y[1, ..., 1:, :], 1, axis=-2
        )[mask]

        # Periodic BC
        self.m_adv_y[0, ..., -1, :] = self.m_adv_y[0, ..., 0, :].copy()

    def nonlinear_flux_correction(self) -> None:
        # for
        pass
