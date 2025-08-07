"""
Advection-diffusion-reaction (ADR) equation solver class in 2D.

partial u / partial t + c * div(u) = lambda u + kappa Laplacian(u)
"""

# from typing import Sequence, Any
from .transport import ScalarTransport, np


class ADR(ScalarTransport):
    """
    Linear ADR solver on a 2D ``Grid``; discretised using the ``CABARET``.
    """

    def __init__(self, nx, ny, left, bottom, right, top, index="std") -> None:
        """
        Main constructor for the ``ADR`` class.
        :return: 2D scalar ADR solver object.
        """

        super().__init__(nx, ny, left, bottom, right, top, index)
        self.m_kappa = 1.0
        # Laplacian stencil with 2 extra ghost cells in each direction, 1 on each side
        self.u = np.empty((self.m_ny + 2, self.m_nx + 2), dtype=np.float64)
        self.laplacian_u = None
        self.laplacian_bcs = None

    @property
    def kappa(self):
        return self.m_kappa

    @kappa.setter
    def kappa(self, value: float):
        self.m_kappa = value

    def second_order_deriv(self):
        # d2udy2
        self.laplacian_u = (
            self.u[2:, 1:-1] - 2.0 * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]
        ) / self.m_dy**2.0

        # d2udx2
        self.laplacian_u += (
            self.u[1:-1, 2:] - 2.0 * self.u[1:-1, 1:-1] + self.u[1:-1, :-2]
        ) / self.m_dx**2.0

    """ Boundary conditions. """

    def periodic_x(self):
        self.u[:, 0] = self.u[:, -2]
        self.u[:, -1] = self.u[:, 1]

    def periodic_y(self):
        self.u[0, :] = self.u[-2, :]
        self.u[-1, :] = self.u[1, :]

    # def dirichlet_west(self):
    #     self.u[:, 0] = 2. * self.m_boundary_values[0] - self.u[:, 1]
    #
    # def dirichlet_south(self):
    #     self.u[0, :] = 2. * self.m_boundary_values[1] - self.u[1, :]
    #
    # def dirichlet_east(self):
    #     self.u[:, -1] = 2. * self.m_boundary_values[2] - self.u[:, -2]
    #
    # def dirichlet_north(self):
    #     self.u[-1, :] = 2. * self.m_boundary_values[3] - self.u[-2, :]
    #
    # def neumann_west(self):
    #     self.u[:, 0] = self.u[:, 1] - self.m_dy * self.m_boundary_values[0]
    #
    # def neumann_south(self):
    #     self.u[0, :] = self.u[1, :] - self.m_dx * self.m_boundary_values[1]
    #
    # def neumann_east(self):
    #     self.u[:, -1] = self.u[:, -2] + self.m_dy * self.m_boundary_values[2]
    #
    # def neumann_north(self):
    #     self.u[-1, :] = self.u[-2, :] + self.m_dx * self.m_boundary_values[3]

    def laplacian_boundary_conditions(self) -> None:
        """
        Enforce Laplacian boundary conditions in the following order: West -> East -> South -> North.
        :return: None
        """

        lbc_funcs = []

        if self.m_boundary_conditions[0] == self.m_boundary_conditions[2] == "Periodic":
            lbc_funcs.append(self.periodic_x)

        else:
            for idx, i in enumerate([0, 2]):  # 0, 1; 0, 2
                bi = (1, -2)[idx]

                if self.m_boundary_conditions[i] == "Dirichlet":

                    def dirichlet_x(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                        self.u[:, tmp_idx] = (
                            2.0 * self.m_boundary_values[tmp_i] - self.u[:, tmp_bi]
                        )

                    lbc_funcs.append(dirichlet_x)

                else:

                    def neumann_x(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                        self.u[:, tmp_idx] = (
                            self.u[:, tmp_bi]
                            + (tmp_idx - tmp_bi)
                            * self.m_dy
                            * self.m_boundary_values[tmp_i]
                        )

                    lbc_funcs.append(neumann_x)

        if self.m_boundary_conditions[1] == self.m_boundary_conditions[3] == "Periodic":
            lbc_funcs.append(self.periodic_y)

        else:
            for idx, i in enumerate([1, 3]):  # 0, 1; 1, 3
                bi = (1, -2)[idx]

                if self.m_boundary_conditions[i] == "Dirichlet":

                    def dirichlet_y(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                        self.u[tmp_idx, :] = (
                            2.0 * self.m_boundary_values[tmp_i] - self.u[tmp_bi, :]
                        )

                    lbc_funcs.append(dirichlet_y)

                else:

                    def neumann_y(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                        self.u[tmp_idx, :] = (
                            self.u[tmp_bi, :]
                            + (tmp_idx - tmp_bi)
                            * self.m_dx
                            * self.m_boundary_values[tmp_i]
                        )

                    lbc_funcs.append(neumann_y)

        self.laplacian_bcs = lambda: [lbc() for lbc in lbc_funcs]

    """ End of boundary conditions. """

    def update_maximum_principle(self):
        self.u[1:-1, 1:-1] = self.solutions[0]

        self.laplacian_bcs()
        self.second_order_deriv()

        self._M_x += self.dt * self.kappa * self.laplacian_u
        self._m_x += self.dt * self.kappa * self.laplacian_u
        self._M_y += self.dt * self.kappa * self.laplacian_u
        self._m_y += self.dt * self.kappa * self.laplacian_u

    def diffusion_dt(self):
        dt_d = (min(self.m_dx, self.m_dy) ** 2) / (4.0 * np.abs(self.kappa))
        self.dt = min(self.dt, dt_d)

    def update_dt_adr(self) -> None:
        self.update_dt(other=self.diffusion_dt)

    def corrector(self):
        self.solutions[0] += self.dt * self.kappa * self.laplacian_u

    def get_pre_processes(self):
        return [
            self.update_dt_adr,
            self.enforce_boundary_conditions,
            self.laplacian_boundary_conditions,
        ]

    def get_solve_sequence(self):
        return [
            self.update_dt_adr,
            self.predictor,
            self.update_maximum_principle,
            self.second_order_extrapolation,
            self.numerical_fluxes,
            self.corrector,
        ]
