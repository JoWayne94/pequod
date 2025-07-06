"""
Abstract base class for a two-dimensional grid class object

Author: JWT
"""
from src.pq_types import *
from abc import ABC, abstractmethod


class Grid(object):
    """
    Base class to represent a 2D uniform grid object in physical space
    """

    def __init__(self, nx, ny, bottom, top, left, right, index) -> None:
        """
        Main constructor

        :param nx:      uint. Number of nodes in the x-direction
        :param ny:      uint. Number of nodes in the y-direction
        :param bottom:  float. y-coordinate of the bottom domain boundary
        :param top:     float. y-coordinate of the top domain boundary
        :param left:    float. x-coordinate of the left domain boundary
        :param right:   float. x-coordinate of the right domain boundary
        :param index:   str. Numbering system of the ``Grid`` object

        :return: 2D ``Grid`` with the given shape.
        """
        self.m_nx = uint32(nx)
        self.m_ny = uint32(ny)
        self.m_bottom = flt64(bottom)
        self.m_top = flt64(top)
        self.m_left = flt64(left)
        self.m_right = flt64(right)
        assert type(index) == str

        self.m_i = range(1, self.m_nx - 1)
        self.m_j = range(1, self.m_ny - 1)
        self.m_x_coords, self.m_dx = np.linspace(self.m_left, self.m_right, self.m_nx, retstep=True, dtype=flt64)
        self.m_y_coords, self.m_dy = np.linspace(self.m_bottom, self.m_top, self.m_ny, retstep=True, dtype=flt64)

        self.m_ndofs = self.m_nx * self.m_ny
        # self.m_ndims = self.m_ndofs**2

        if index == 'std':
            self.m_labels = self.std_indexing()
        elif index == 'onion':
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
        Return labels of grid nodes in an onion indexed fashion
        :return: ``labels`` numpy array
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
        Return labels of grid nodes in the default way
        :return: ``labels`` numpy array
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

