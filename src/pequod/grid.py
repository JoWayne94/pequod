"""
Data class for a two-dimensional grid class object.

Todo: 1. Implement setters and getters for all parameters
"""

from .pq_types import *

# from dataclasses import dataclass
# from typing import Iterable, Iterator, Self, Sequence, overload


class Grid(object):
    """
    Base class to represent a 2D uniform grid object in physical space.
    """

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        west: float = 0.0,
        south: float = 0.0,
        east: float = 1.0,
        north: float = 1.0,
        index: str = "std",
    ) -> None:
        """
        Main constructor for the computational grid.

        Parameters
        ----------
        nx : uint
                Number of advection-type nodes in the x-direction.
        ny : uint
                Number of advection-type nodes in the y-direction.
        west : float
                y-coordinate of the west domain boundary.
        south : float
                y-coordinate of the south domain boundary.
        east : float
                x-coordinate of the east domain boundary.
        north : float
                x-coordinate of the north domain boundary.
        index : str
                Numbering system of the ``Grid`` object.

        Returns
        -------
        None
                2D ``Grid`` with the given shape.
        """
        self.m_nx_adv = uint32(nx)
        self.m_ny_adv = uint32(ny)
        self.m_west = flt64(west)
        self.m_south = flt64(south)
        self.m_east = flt64(east)
        self.m_north = flt64(north)

        self.m_nx = self.m_nx_adv - 1
        self.m_ny = self.m_ny_adv - 1

        self.m_i = range(1, self.m_nx - 1)
        self.m_j = range(1, self.m_ny - 1)
        self.m_x_coords_adv, self.m_dx = np.linspace(
            self.m_west, self.m_east, self.m_nx_adv, retstep=True, dtype=flt64
        )
        self.m_y_coords_adv, self.m_dy = np.linspace(
            self.m_south, self.m_north, self.m_ny_adv, retstep=True, dtype=flt64
        )

        self.m_x_coords = self.m_x_coords_adv[:-1] + 0.5 * self.m_dx
        self.m_y_coords = self.m_y_coords_adv[:-1] + 0.5 * self.m_dy

        self.m_ndofs = self.m_nx * self.m_ny
        # self.m_ndims = self.m_ndofs**2

        if index == "std":
            self.m_labels = self.std_indexing()
        elif index == "onion":
            self.m_labels = self.onion_indexing()
        else:
            NotImplementedError("Numbering system not implemented.")

    @property
    def get_x_coords(self) -> np.array:
        return self.m_x_coords

    @property
    def get_y_coords(self) -> np.array:
        return self.m_y_coords

    def onion_indexing(self) -> np.ndarray:
        """
        Return labels of grid nodes in an onion indexed fashion.
        :return: ``labels`` numpy array.
        """

        labels = np.empty((self.m_ny, self.m_nx), dtype=uint32)

        counter = 0
        # s = i + j, which ranges from 0 up to (nx + ny - 2)
        for s in range(self.m_nx + self.m_ny - 1):
            # i goes from 0 up to s, and j = s - i
            for i in range(s + 1):
                j = s - i
                # Check if we are within the bounds of the grid
                if 0 <= i < self.m_ny and 0 <= j < self.m_nx:
                    # Assign the counter to the grid at row = i, col = j.
                    # Note: i = 0 is the bottom row in this convention.
                    labels[i, j] = counter
                    counter += 1

        # By default, row index i = 0 is at the top of the logical grid. To put row 0 at bottom, flip vertically:
        labels_flipped = np.flipud(labels)

        return labels_flipped

    def std_indexing(self) -> np.ndarray:
        """
        Return labels of grid nodes in the default way.
        :return: ``labels`` numpy array.
        """

        labels = np.empty((self.m_ny, self.m_nx), dtype=uint32)

        num = 0  # Start numbering from 0
        for i in range(self.m_ny):
            for j in range(self.m_nx):
                labels[self.m_ny - i - 1, j] = num
                num += 1

        # By default, row index i = 0 is at the bottom of the logical grid. To put row 0 at the top, flip vertically:
        labels_flipped = np.flipud(labels)

        return labels

    @property
    def labels(self) -> np.ndarray:
        return self.m_labels
