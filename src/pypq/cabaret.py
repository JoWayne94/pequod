"""
The CABARET on a 2D ``Grid``

Author: JWT
"""
from typing import Callable
from src.grid import *


class CABARET(Grid):
    """
    Second-order upwind leapfrog on a 2D uniform ``Grid``
    """

    def __init__(self, nx, ny, bottom, top, left, right, index='std') -> None:
        """
        Main constructor

        :param nx:      uint. Number of nodes in the x-direction
        :param ny:      uint. Number of nodes in the y-direction
        :param bottom:  float. y-coordinate of the bottom domain boundary
        :param top:     float. y-coordinate of the top domain boundary
        :param left:    float. x-coordinate of the left domain boundary
        :param right:   float. x-coordinate of the right domain boundary
        :param index:   str. Numbering system of the ``Grid`` object

        :return: 2D CABARET derived from ``Grid`` with the given shape.
        """

        super().__init__(nx, ny, bottom, top, left, right, index)
        self.m_nx_adv = self.m_nx - 1
        self.m_ny_adv = self.m_ny - 1
        self.m_nvars = 1
        self.m_dt = 1.

        # Boundary value problem - default settings are doubly periodic
        self.m_boundary_conditions = ['Periodic', 'Periodic', 'Periodic', 'Periodic']
        self.m_boundary_values = [None, None, None, None]

        # Conservative-type variables
        self.m_cons = None

        # Advection-type variables
        self.m_adv_x = None
        self.m_adv_y = None

        self.m_x_coords_adv = self.m_x_coords[:-1] + 0.5 * self.m_dx
        self.m_y_coords_adv = self.m_y_coords[:-1] + 0.5 * self.m_dy


    @property
    def nvars(self):
        """
        Getter for the number of variables in equations.
        :return: Number of variables
        """
        return self.m_nvars

    @nvars.setter
    def nvars(self, value: uint8):
        """
        Setter for the number of variables in equations.
        """
        self.m_nvars = value

    @property
    def solutions(self):
        """
        Getter for solution values.
        :return: Conservative-type solutions
        """
        return self.m_cons

    """ BVP methods """

    def bcs(self):

    def boundary_conditions(self, bcs: list) -> None:

        assert len(bcs) == 4
        tmp = [2, 3, 0, 1]

        for i, bc in enumerate(bcs):

            assert type(bc) == str

            if bc == 'Periodic' and bcs[tmp[i]] != 'Periodic':
                raise ValueError(f"If boundary {i} is Periodic, then boundary {tmp[i]} must also be Periodic. "
                                 f"Else, please change condition of boundary {i}.")



        # Conservative-type variables
        self.m_cons = np.zeros((self.m_nvars, self.m_ny, self.m_nx))

        # Advection-type variables
        self.m_adv_x = np.zeros((self.m_nvars, self.m_ny, self.m_nx_adv))
        self.m_adv_y = np.zeros((self.m_nvars, self.m_ny_adv, self.m_nx))

    """ End of BVP methods """

    def set_initial_conditions(self, ic: Callable = None, var: int = 0) -> None:

        # Initial datum/conditions defined
        if ic is not None:
            for i in np.arange(self.m_nx):
                for j in np.arange(self.m_ny):

                    self.m_cons[var][j][i] = ic(self.m_x_coords[i], self.m_y_coords[j])

            for i in np.arange(self.m_nx_adv):
                for j in np.arange(self.m_ny):

                    self.m_adv_x[var][j][i] = ic(self.m_x_coords_adv[i], self.m_y_coords[j])

            for i in np.arange(self.m_nx):
                for j in np.arange(self.m_ny_adv):

                    self.m_adv_y[var][j][i] = ic(self.m_x_coords[i], self.m_y_coords_adv[j])

    def predictor(self, flux_values: np.ndarray) -> np.ndarray:

        # assert


