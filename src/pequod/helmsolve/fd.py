"""
Child class of a 2D grid object discretised using centred finite differencing

Author: JWT
"""
import numpy as np

from grid import Grid


class FD2(Grid):
    """
    Second-order finite differences on a 2D uniform ``Grid``
    """

    def __init__(self, nx, ny, bottom, top, left, right, index='std', dbcs='lhs'):
        """
        Main constructor

        :param nx:      np.uint8. Number of nodes in the x-direction
        :param ny:      np.uint8. Number of nodes in the y-direction
        :param bottom:  np.float64. y-coordinate of the bottom domain boundary
        :param top:     np.float64. y-coordinate of the top domain boundary
        :param left:    np.float64. x-coordinate of the left domain boundary
        :param right:   np.float64. x-coordinate of the right domain boundary
        :param index:   str. Numbering system of the ``Grid`` object
        :param dbcs:    str. Determine if Dirichlet boundary values are on
                        the left or right hand side of the linear system

        :return: 2D finite (centred) difference ``Grid`` with the given shape.
        """

        super().__init__(nx, ny, bottom, top, left, right, index)
        self.m_k_squared = 1.

        self.mass_matrix = self.zeroth_derivative_matrix()
        self.laplacian_matrix, self.forcing_vector = self.second_derivative_matrix()
        self.helmholtz_operator = self.mass_matrix + self.laplacian_matrix



    @property
    def k_squared(self):
        """
        Getter for the wave number squared (eigenvalue) of the equation.
        :return: ``k**2``
        """
        return self.m_k_squared

    @k_squared.setter
    def k_squared(self, value):
        """
        Setter for the wave number squared (eigenvalue) of the equation.
        """
        self.m_k_squared = value

    def zeroth_derivative_matrix(self):
        """
        Returns the global mass matrix of the system

        :return:
        """
        return self.m_k_squared * np.eye(self.m_ndofs, dtype=np.float64)

    def first_derivative_matrix(self):
        pass

    def second_derivative_matrix(self):
        """
        Returns the global Laplacian matrix of the system

        :return:
        """
        laplacian_matrix = np.zeros((self.m_ndofs, self.m_ndofs), dtype=np.float64)
        forcing_vector = np.empty(self.m_ndofs, dtype=np.float64)

        # print(self.m_labels[1:self.m_nx - 1, 1:self.m_ny - 1])
        for i in self.m_i:
            for j in self.m_j:
                row = self.labels[i, j]
                up = self.labels[i, j + 1]
                down = self.labels[i, j - 1]
                left = self.labels[i - 1, j]
                right = self.labels[i + 1, j]

                # row = i + j * N  # Flattened index

                # Main diagonal
                laplacian_matrix[row, row] = 4. / (self.m_dx * self.m_dy)

                # Left neighbour: (i-1, j)
                laplacian_matrix[row, left] = -1. / (self.m_dx**2)

                # Right neighbour: (i+1, j)
                laplacian_matrix[row, right] = -1. / (self.m_dx**2)

                # Down neighbour: (i, j-1)
                laplacian_matrix[row, down] = -1. / (self.m_dy**2)

                # Up neighbor: (i, j+1)
                laplacian_matrix[row, up] = -1. / (self.m_dy**2)

                forcing_vector[row] = 2. * np.pi**2 * np.sin(np.pi * self.m_x_coords[i]) * np.sin(np.pi * self.m_y_coords[j])

        """ Dirichlet nodes """
        # Bottom boundary
        for n in self.labels[-1]:
            laplacian_matrix[n, n] = 1.
            forcing_vector[n] = 0.

        # Top boundary
        for n in self.labels[0]:
            laplacian_matrix[n, n] = 1.
            forcing_vector[n] = 0.

        # Left boundary
        for n in self.labels[:, 0]:
            laplacian_matrix[n, n] = 1.
            forcing_vector[n] = 0.

        # Right boundary
        for n in self.labels[:, -1]:
            laplacian_matrix[n, n] = 1.
            forcing_vector[n] = 0.

        return laplacian_matrix, forcing_vector
