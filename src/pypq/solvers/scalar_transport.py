"""
Two-dimensional scalar advection solver class

\partial u / \ partial t + div(F) = 0

Author: JWT
"""
from typing import Callable
from src.solvers.solver import Solver


class ScalarTransport(Solver):
    """
    Scalar advection solver; child class of a 2D ``Grid`` discretised using the ``CABARET``.
    """

    def __init__(self, nx, ny, bottom, top, left, right, index='std', adv_vel=None) -> None:
        """
        Main constructor

        :param nx:      uint. Number of nodes in the x-direction
        :param ny:      uint. Number of nodes in the y-direction
        :param bottom:  float. y-coordinate of the bottom domain boundary
        :param top:     float. y-coordinate of the top domain boundary
        :param left:    float. x-coordinate of the left domain boundary
        :param right:   float. x-coordinate of the right domain boundary
        :param index:   str. Numbering convention of the ``Grid`` object
        :param adv_vel: [float, float] or a Python method. Advective velocity

        :return: 2D scalar transport solver class object.
        """

        super().__init__(nx, ny, bottom, top, left, right, index)
        if adv_vel is None:
            adv_vel = [1., 0.]
        self.m_adv_vel = adv_vel

    @property
    def adv_vel(self):
        """
        Getter for advective velocity.
        :return: Advective velocity
        """
        return self.m_adv_vel

    @adv_vel.setter
    def adv_vel(self, value):
        """
        Setter for advective velocity.
        """
        if len(value) != 2 or type(value) != Callable:
            raise TypeError('Advective velocity must be a callable object or [v_x, v_y].')
        self.m_adv_vel = value

    def flux_def(self, u):
        """
        Flux function for the transport equation.

        :param u:  float or np.ndarray. Advected scalar quantity

        :return: F = c(x, y; t) * u
        """
        return self.m_adv_vel * u

    def numerical_flux(self):
        """
        Numerical flux for the transport equation.
        :return:
        """

    def solve(self) -> None:

        pass