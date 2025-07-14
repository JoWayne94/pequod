"""
The CABARET on a 2D ``Grid``.

Todo: 1) Modify BCs to be functions instead of if statements
"""

from typing import Callable, Self

from .grid import *


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

        # Boundary value problem (BVP) - default settings are doubly periodic
        self.m_boundary_conditions = ["Periodic", "Periodic", "Periodic", "Periodic"]
        self.m_boundary_values = [None, None, None, None]

        # Conservative-type data
        self.m_cons = np.zeros((self.m_nvars, self.m_ny, self.m_nx))

        # Advection-type data
        self.m_adv_x = np.zeros((2, self.m_nvars, self.m_ny, self.m_nx_adv))
        self.m_adv_y = np.zeros((2, self.m_nvars, self.m_ny_adv, self.m_nx))

        self.m_adv_vel = None

        self.X, self.Y = np.meshgrid(self.get_x_coords, self.get_y_coords)
        self.X_adv_x, self.Y_adv_x = np.meshgrid(self.m_x_coords_adv, self.get_y_coords)
        self.X_adv_y, self.Y_adv_y = np.meshgrid(self.get_x_coords, self.m_y_coords_adv)

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
        :return: Time step size.
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

    def enforce_boundary_conditions(self) -> None:
        """
        Enforce boundary conditions in the following order: West -> East -> South -> North.
        :return: None
        """

        # Periodic BCs with upwind flux
        if self.m_boundary_conditions[0] == self.m_boundary_conditions[2] == "Periodic":
            mask = self.m_adv_vel[0][..., 0] > 0.0
            self.m_adv_x[0, ..., 0][mask] = self.m_adv_x[1, ..., -1][mask].copy()
            # self.m_adv_x[0, ..., -1][~mask] = self.m_adv_x[1, ..., 0][~mask].copy()
            self.m_adv_x[0, ..., -1] = self.m_adv_x[0, ..., 0].copy()

        else:
            for i in (0, 2):
                bi = -(i != 0)

                if self.m_boundary_conditions[i] == "Dirichlet":
                    self.m_adv_x[0, ..., bi] = self.m_boundary_values[i] * np.ones(
                        self.m_ny
                    )
                else:
                    continue

        if self.m_boundary_conditions[1] == self.m_boundary_conditions[3] == "Periodic":
            mask = self.m_adv_vel[1][..., 0, :] > 0.0
            self.m_adv_y[0, ..., 0, :][mask] = self.m_adv_y[1, ..., -1, :][mask].copy()
            # self.m_adv_y[0, ..., -1, :][~mask] = self.m_adv_y[1, ..., 0, :][~mask].copy()
            self.m_adv_y[0, ..., -1, :] = self.m_adv_y[0, ..., 0, :].copy()

        else:
            for i in (1, 3):
                bi = -(i - 1 != 0)

                if self.m_boundary_conditions[i] == "Dirichlet":
                    self.m_adv_y[0, ..., bi, :] = self.m_boundary_values[i] * np.ones(
                        self.m_nx
                    )
                else:
                    continue

    """ End of BVP methods and start of CABARET routines. """

    def set_advection_velocity(
        self, value: tuple[float | np.ndarray, float | np.ndarray]
    ):
        # if type(value[0] == np.ndarray):
        #     assert value[0].ndim == 2
        #     assert value[0].shape == (self.m_nx_adv, self.m_ny)
        # if type(value[1] == np.ndarray):
        #     assert value[1].ndim == 2
        #     assert value[1].shape == (self.m_nx, self.m_ny_adv)

        self.m_adv_vel = (
            np.expand_dims(np.asarray(value[0], dtype=flt64).T, 0),
            np.expand_dims(np.asarray(value[1], dtype=flt64).T, 0),
        )

    def adv_to_flux(self):
        """
        Convert advection-type data to flux values.
        :return:
        """
        return self.m_adv_x[0] * self.m_adv_vel[0], self.m_adv_y[0] * self.m_adv_vel[1]

    def predictor_corrector(self) -> None:
        f_x, f_y = self.adv_to_flux()

        self.m_cons += (
            0.5
            * self.dt
            * (
                (-f_x[..., 1:] + f_x[..., :-1]) / self.m_dx
                + (-f_y[..., 1:, :] + f_y[..., :-1, :]) / self.m_dy
            )
        )

    def second_order_extrapolation(self) -> None:
        # Extrapolate from the west/south
        self.m_adv_x[1, ..., 1:] = (
            2.0 * self.m_cons - self.m_adv_x[0, ..., :-1]
        )  # .copy()
        self.m_adv_y[1, ..., 1:, :] = 2.0 * self.m_cons - self.m_adv_y[0, ..., :-1, :]

        # Extrapolate from the east/north
        self.m_adv_x[0, ..., :-1] = 2.0 * self.m_cons - self.m_adv_x[0, ..., 1:]
        self.m_adv_y[0, ..., :-1, :] = 2.0 * self.m_cons - self.m_adv_y[0, ..., 1:, :]

    def nonlinear_flux_correction(self) -> None:
        pass

    def numerical_flux(self) -> None:
        """
        Upwind flux calculations.
        :return: None
        """

        # Update in x-direction
        mask = self.m_adv_vel[0][..., 1:-1] > 0.0
        self.m_adv_x[0, ..., 1:-1][mask] = self.m_adv_x[1, ..., 1:-1][mask]  # .copy()

        # Update in y-direction
        mask = self.m_adv_vel[1][..., 1:-1, :] > 0.0
        self.m_adv_y[0, ..., 1:-1, :][mask] = self.m_adv_y[1, ..., 1:-1, :][mask]
