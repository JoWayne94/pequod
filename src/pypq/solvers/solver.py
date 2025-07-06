"""
Base class to implement different types of physics solvers

Author: JWT
"""
from src.cabaret import *
from abc import ABC, abstractmethod


class Solver(ABC, CABARET):
    """
    Solver base class for various physics extensions
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

        :return: 2D physics-based solver equipped with the ``CABARET`` advector, given grid dimensions.
        """

        super().__init__(nx, ny, bottom, top, left, right, index)


    @abstractmethod
    def flux_def(self, *args, **kwargs):
        """
        Insert definitions of conservative flux functions
        :return:
        """
        pass

    @abstractmethod
    def numerical_flux(self):
        """
        Define specific numerical flux calculations
        :return:
        """
        pass
