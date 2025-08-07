"""
Validation test cases for ``transport`` class.

Configure solvers with:
[nx, ny, west, south, east, north, (west_bc, south_bc, east_bc, north_bc),
 (west_bv, south_bv, east_bv, north_bv), (vx, vy), cfl, final_time]
"""

import numpy as np


class OneDimCases:
    """
    1D test cases.
    """

    def gaussian(self, x, y):
        center = [0.0, 0.0]
        std = 0.15
        return (
            np.exp(
                (-1.0 / (2.0 * std**2)) * ((x - center[0]) ** 2 + (y - center[1]) ** 2)
            )
            if y == 0.0
            else 0.0
        )

    def case_gaussian(self):
        """
        A parametrised Gaussian travelling wave.
        """
        conf = [
            65,
            2,
            -1.0,
            -1.0,
            1.0,
            1.0,
            ("Periodic", "Dirichlet", "Periodic", "Dirichlet"),
            (None, 0.0, None, 0.0),
            (1.0, 0.0),
            0.5,
            2.0,
        ]

        return conf, self.gaussian

    # def case_tophat(self):
    #     """
    #     A top hat signal.
    #     """
    #     conf = [1025, 2, 0., -1., 2., 1., ('Periodic', 'Dirichlet', 'Periodic', 'Dirichlet'),
    #             (None, 0., None, 0.), (2., 0.), 0.5, 1.]
    #
    #     ic = lambda x, y: np.where((0.5 <= x) & (x <= 1.5) & (y == 0.), 1., 0.)
    #
    #     return conf, ic


class ConvergenceCases:
    conf = [
        33,
        2,
        0.0,
        -1.0,
        2.0 * np.pi,
        1.0,
        ("Periodic", "Dirichlet", "Periodic", "Dirichlet"),
        (None, 0.0, None, 0.0),
        (1.0, 0.0),
        0.8,
        2.0 * np.pi,
    ]

    def sine(self, x, y):
        """
        A sine wave.
        """
        return np.sin(x) if y == 0.0 else 0.0

    case = sine

    def case_32(self):
        """
        N_x = 32
        """
        conf = self.conf

        return conf, self.case

    def case_64(self):
        """
        N_x = 64
        """
        conf = self.conf
        conf[0] = 65

        return conf, self.case

    def case_128(self):
        """
        N_x = 128
        """
        conf = self.conf
        conf[0] = 129

        return conf, self.case

    def case_256(self):
        """
        N_x = 256
        """
        conf = self.conf
        conf[0] = 257

        return conf, self.case


class TwoDimCases:
    """
    2D test cases.
    """

    class GaussianBlob:
        """
        A parametrised Gaussian blob.
        """

        conf = [
            129,
            129,
            0.0,
            0.0,
            1.0,
            1.0,
            ("Periodic", "Periodic", "Periodic", "Periodic"),
            (None, None, None, None),
            (0.0, 1.0),
            0.5,
            1.0,
        ]

        def gaussian(self, x, y):
            center = [0.5, 0.5]
            std = 0.15
            return np.exp(
                (-1.0 / (2.0 * std**2)) * ((x - center[0]) ** 2 + (y - center[1]) ** 2)
            )

        case = gaussian

        def case_up(self):
            """
            North.
            """
            conf = self.conf

            return conf, self.case

        def case_down(self):
            """
            South.
            """
            conf = self.conf
            conf[8] = (0.0, -1.0)

            return conf, self.case

        def case_left(self):
            """
            West.
            """
            conf = self.conf
            conf[8] = (-1.0, 0.0)

            return conf, self.case

        def case_right(self):
            """
            East.
            """
            conf = self.conf
            conf[8] = (1.0, 0.0)

            return conf, self.case

        def case_diag(self):
            """
            North-east.
            """
            conf = self.conf
            conf[8] = (1.0, 1.0)

            return conf, self.case
