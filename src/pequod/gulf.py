"""
The CABARET on a 2D ``Grid``.

Todo: 1) Write getters for solutions.
      2) Channel and basin modes.
"""

from typing import Any, Sequence
from .grid import Grid, np, uint8
from .pq_types import eps64


class Gulf:
    """
    Second-order Generalised Upwind LeapFrog (GULF-2) on a 2D uniform ``Grid``.
    """

    def __init__(self, domain: Grid) -> None:
        """
        Main constructor for GULF.
        :return: 2D CABARET derived from ``Grid`` with the given shape.
        """

        # Domain properties
        self.domain = domain

        self._nx_adv = self.domain.nx
        self._ny_adv = self.domain.ny

        self._x_faces = self.domain.x_coords_adv
        self._y_faces = self.domain.y_coords_adv

        self._nvars = 1
        self._dt = 1.0

        self.X, self.Y = np.meshgrid(self.domain.x_coords, self.domain.y_coords)
        self.X_adv_x, self.Y_adv_x = np.meshgrid(self._x_faces, self.domain.y_coords[:-1])
        self.X_adv_y, self.Y_adv_y = np.meshgrid(self.domain.x_coords[:-1], self._y_faces)

        # Numerics
        self.predictor = self.predictor_corrector(self._predictor)
        self.corrector = self.predictor_corrector(self._corrector)

        self._init_solutions_data()

    @property
    def nx_faces(self):
        """
        Number of cell faces in the x-direction.
        :return: n_x
        """
        return self._nx_adv

    @property
    def ny_faces(self):
        """
        Number of cell faces in the y-direction.
        :return: n_y
        """
        return self._ny_adv

    @property
    def x_coords_faces(self) -> np.ndarray:
        """
        :return: Cell faces x-coordinates.
        """
        return self._x_faces

    @property
    def y_coords_faces(self) -> np.ndarray:
        """
        :return: Cell faces y-coordinates.
        """
        return self._y_faces

    def _init_solutions_data(self):

        # Conservative-type data
        self._cons = np.zeros((self.n_vars, self.domain.ny, self.domain.nx))
        self._Q_x, self._Q_y, self._Q = None, None, None

        # Advection-type data
        self._adv_x = np.zeros((2, self.n_vars, self.domain.ny, self.nx_faces))
        self._adv_y = np.zeros((2, self.n_vars, self.ny_faces, self.domain.nx))
        self._adv_vel = None

    @property
    def n_vars(self):
        """
        Getter for the number of variables in equations.
        :return: Number of variables.
        """
        return self._nvars

    @n_vars.setter
    def n_vars(self, value: int):
        """
        Setter for the number of variables in equations.
        """
        self._nvars = uint8(value)
        self._init_solutions_data()

    @property
    def dt(self):
        """
        Getter for the time-step size.
        :return: Time-step size.
        """
        return self._dt

    @dt.setter
    def dt(self, value: float):
        """
        Setter for the time-step size.
        """
        self._dt = value

    @property
    def solutions(self):
        """
        Getter for solution values.
        :return: Conservative-type data.
        """
        return self._cons

    """ BVP methods. """

    def _boundary_conditions(
        self,
        periodic_x: Sequence[Any] = None,
        periodic_y: Sequence[Any] = None,
        internal_x: Sequence[Any] = None,
        internal_y: Sequence[Any] = None,
        dirichlet: tuple[
            Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]
        ] = None,
    ) -> None:
        """
        Enforce boundary conditions in the following order: West -> East -> South -> North.
        :return: None
        """

        self.flux_funcs = []

        for bc, bv in self.domain.boundary_conditions.items():
            if bc == "Periodic":

                if periodic_x is not None:
                    self.flux_funcs.append(*periodic_x)
                if periodic_y is not None:
                    self.flux_funcs.append(*periodic_y)

            if bc == "Channel" or bc == "Basin":

                if periodic_x is not None:
                    self.flux_funcs.append(*periodic_x)

                if internal_y is not None:
                    self.flux_funcs.append(*internal_y)
                # if dirichlet is not None:
                #     self.flux_funcs.append(*dirichlet[i])

                # def dirichlet_y(val=bv) -> None:
                #     # South
                #     self._adv_x[0, ..., 0, :] = val[1] * np.ones(self.domain.nx)
                #     self._cons[0, 0, :] = val[1] * np.ones(self.domain.nx)
                #     # First-order extrapolation from the South/North
                #     self._adv_y[1, ..., 0, :] = val[1] * np.ones(self.domain.nx)
                #     self._adv_y[0, ..., -1, :] = val[3] * np.ones(self.domain.nx)

                # self.flux_funcs.append(dirichlet_y)

            if bc == "Basin":

                if internal_x is not None:
                    self.flux_funcs.append(*internal_x)
                # if dirichlet is not None:
                #     self.flux_funcs.append(*dirichlet[i])

                # def dirichlet_x(val=bv) -> None:
                #     # West
                #     self._adv_y[0, ..., 0] = val[0] * np.ones(self.domain.ny)
                #     self._cons[0, :, 0] = val[0] * np.ones(self.domain.ny)
                #     # First-order extrapolation from the West/East
                #     self._adv_x[1, ..., 0] = val[0] * np.ones(self.domain.ny)
                #     self._adv_x[0, ..., -1] = val[2] * np.ones(self.domain.ny)

                # self.flux_funcs.append(dirichlet_x)

                # if internal_y is not None:
                #     self.flux_funcs.append(*internal_y)
                # self.flux_funcs.append(self.dirichlet_y)

            # else:
            #     raise ValueError(" BCs not recognised in Gulf object. ")

        #         elif self.domain.boundary_conditions[boundaries[i]][0] == "Neumann":
        #             if neumann is not None:
        #                 self.flux_funcs.append(*neumann[i])
        #             if i == 0:
        #                 def neumann_x():
        #                     mask = self._adv_vel[0][..., 0] > 0.0
        #                     self._adv_x[0, ..., 0][mask] = self._adv_x[1, ..., 1][
        #                         mask
        #                     ]
        #             else:
        #                 def neumann_x():
        #                     self._adv_x[0, ..., -1] = self._adv_x[1, ..., -1].copy()
        #                     mask = self._adv_vel[0][..., -1] < 0.0
        #                     self._adv_x[0, ..., -1][mask] = self._adv_x[0, ..., -2][
        #                         mask
        #                     ]
        #             self.flux_funcs.append(neumann_x)
            # for i in (1, 3):
            #     bi = (i == 1) - 1
            #     elif self.domain.boundary_conditions[boundaries[i]][0] == "Neumann":
            #         if neumann is not None:
            #             self.flux_funcs.append(*neumann[i])
            #         if i == 1:
            #             def neumann_y():
            #                 mask = self._adv_vel[1][..., 0, :] > 0.0
            #                 self._adv_y[0, ..., 0, :][mask] = self._adv_y[
            #                     1, ..., 1, :
            #                 ][mask]
            #         else:
            #             def neumann_y():
            #                 self._adv_y[0, ..., -1, :] = self._adv_y[
            #                     1, ..., -1, :].copy()
            #                 mask = self._adv_vel[1][..., -1, :] < 0.0
            #                 self._adv_y[0, ..., -1, :][mask] = self._adv_y[
            #                     0, ..., -2, :
            #                 ][mask]
            #         self.flux_funcs.append(neumann_y)

        self.flux_funcs.append(self.numerical_flux_x)
        self.flux_funcs.append(self.numerical_flux_y)
        self.numerical_fluxes = lambda: [update_flux() for update_flux in self.flux_funcs]

    """ End of BVP methods and start of CABARET routines. """

    def physical_flux(self):
        """
        Template of physical fluxes for subclass solvers.
        :return: Convert advection-type data to flux values.
        """
        return self._adv_x[0], self._adv_y[0]

    def predictor_corrector(self, func):
        """
        Update conservative-typed values.
        :param func: Any procedures after half or full time-step update.
        :return:     Predictor or corrector steps.
        """
        def wrapper() -> None:
            f_x, f_y = self.physical_flux()

            # Data can be rolled even if non-periodic as long as BCs enforced
            # straight afterwards
            self._Q_x = (np.roll(f_x, 1, axis=-1) - f_x) / self.domain.dx
            self._Q_y = (np.roll(f_y, 1, axis=-2) - f_y) / self.domain.dy
            self._Q = self._Q_x + self._Q_y

            self._cons += 0.5 * self.dt * self._Q

            func()

        return wrapper

    def _predictor(self):
        """
        Compute M and m values required for nonlinear flux correction after the predictor step.
        :return: None
        """
        self._M_x = np.maximum.reduce(
            [self._adv_x[0], self._cons, np.roll(self._adv_x[0], 1, axis=-1)]
        )
        self._m_x = np.minimum.reduce(
            [self._adv_x[0], self._cons, np.roll(self._adv_x[0], 1, axis=-1)]
        )
        self._M_y = np.maximum.reduce(
            [self._adv_y[0], self._cons, np.roll(self._adv_y[0], 1, axis=-2)]
        )
        self._m_y = np.minimum.reduce(
            [self._adv_y[0], self._cons, np.roll(self._adv_y[0], 1, axis=-2)]
        )
        # Again, results unaffected by rolling as boundary data overwritten if non-periodic

        self._M_x += self.dt * self._Q_y
        self._m_x += self.dt * self._Q_y
        self._M_y += self.dt * self._Q_x
        self._m_y += self.dt * self._Q_x

    def _corrector(self):
        pass

    def second_order_extrapolation(self) -> None:
        # Extrapolate from the west/south
        self._adv_x[1] = 2.0 * self._cons - np.roll(self._adv_x[0], 1, axis=-1)
        self._adv_y[1] = 2.0 * self._cons - np.roll(self._adv_y[0], 1, axis=-2)

        # Extrapolate from the east/north
        self._adv_x[0] = np.roll(2.0 * self._cons - self._adv_x[0], -1, axis=-1)
        self._adv_y[0] = np.roll(2.0 * self._cons - self._adv_y[0], -1, axis=-2)

        # Upper and lower bounds from the maximum principle
        np.clip(
            self._adv_x[1], self._m_x + eps64, self._M_x - eps64, out=self._adv_x[1]
        )
        np.clip(
            self._adv_x[0], np.roll(self._m_x, -1, axis=-1) + eps64, np.roll(self._M_x, -1, axis=-1) - eps64,
            out=self._adv_x[0]
        )
        np.clip(
            self._adv_y[1], self._m_y + eps64, self._M_y - eps64, out=self._adv_y[1]
        )
        np.clip(
            self._adv_y[0], np.roll(self._m_y, -1, axis=-2) + eps64, np.roll(self._M_y, -1, axis=-2) - eps64,
            out=self._adv_y[0]
        )

    """ Numerical flux methods. """

    # def numerical_flux_x(self) -> None:
    #     """
    #     Upwind flux calculations for internal cell faces in the x-direction.
    #     :return: None
    #     """
    #     # Update in x-direction
    #     mask = self._adv_vel[0][..., 1:-1] > 0.0
    #     self._adv_x[0, ..., 1:-1][mask] = self._adv_x[1, ..., 1:-1][mask]
    # def numerical_flux_y(self) -> None:
    #     """
    #     Upwind flux calculations for internal cell faces in the y-direction.
    #     :return: None
    #     """
    #     # Update in y-direction
    #     mask = self._adv_vel[1][..., 1:-1, :] > 0.0
    #     self._adv_y[0, ..., 1:-1, :][mask] = self._adv_y[1, ..., 1:-1, :][mask]

    def numerical_flux_x(self) -> None:
        """
        Upwind flux calculations in the x-direction with periodic boundary conditions.
        :return: None
        """
        # Update in x-direction
        mask = self._adv_vel[0] > 0.0
        # self._adv_x[0, ...][mask] = np.roll(self._adv_x[1, ...], 1, axis=-1)[mask]
        self._adv_x[0, ...][mask] = self._adv_x[1, ...][mask]

    def numerical_flux_y(self) -> None:
        """
        Upwind flux calculations in the y-direction with periodic boundary conditions.
        :return: None
        """
        # Update in y-direction
        mask = self._adv_vel[1] > 0.0
        # self._adv_y[0, ...][mask] = np.roll(self._adv_y[1, ...], 1, axis=-2)[mask]
        self._adv_y[0, ...][mask] = self._adv_y[1, ...][mask]
