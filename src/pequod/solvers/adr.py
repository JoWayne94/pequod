"""
Two-dimensional advection-diffusion-reaction (ADR) equation solver class.

partial u / partial t + c * div(u) = lambda u + kappa Laplacian(u)
"""

from .transport import ScalarTransport


class ADR(ScalarTransport):
    """
    Linear ADR solver on a 2D ``Grid``; discretised using the ``CABARET``.
    """

    def __init__(self, nx, ny, bottom, top, left, right, index="std") -> None:
        """
        Main constructor for the ``ADR`` class.
        :return: 2D scalar ADR solver object.
        """

        super().__init__(nx, ny, bottom, top, left, right, index)

    def get_pre_processes(self):
        return [self.update_dt, self.enforce_boundary_conditions]

    def get_solve_sequence(self):
        return [
            self.update_dt,
            self.predictor,
            self.second_order_extrapolation,
            self.numerical_fluxes,
            self.corrector,
        ]
