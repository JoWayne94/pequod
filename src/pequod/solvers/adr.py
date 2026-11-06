"""
Advection-diffusion-reaction (ADR) equation solver class in 2D.

partial u / partial t + c * div u = lambda * u + nu * nabla^2 u

Todo: 1) Check if Laplacian source terms are required in the maximum principles.
"""

from functools import partial
from .transport import Grid, ScalarTransport, np


class ADR(ScalarTransport):
    """
    Linear ADR solver on a 2D ``Grid``; discretised using the ``CABARET``.
    """

    def __init__(self, domain: Grid) -> None:
        """
        Main constructor for the ``ADR`` class.
        :return: 2D scalar ADR solver object.
        """

        super().__init__(domain)
        self._nu = 1.0
        self.dt_d = self.get_diffusion_timestep

        # Laplacian stencil with 2 extra ghost points in each direction, 1 on each side
        self.u = np.empty((self.domain.ny + 2, self.domain.nx + 2), dtype=np.float64)

    @property
    def get_diffusion_timestep(self):
        if self.nu == 0.0:
            return np.inf
        return (
            min(self.cfl, 0.5)
            * min(self.domain.dx, self.domain.dy) ** 2
            / (2.0 * self.nu)
        )

    @property
    def nu(self):
        """
        :return: Viscosity.
        """
        return self._nu

    @nu.setter
    def nu(self, value: float):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("nu must be finite and non-negative.")
        self._nu = value
        self.dt_d = self.get_diffusion_timestep

    def _second_order_deriv(self) -> None:
        # d2udy2
        self.laplacian_u = (
            self.u[2:, 1:-1] - 2.0 * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]
        ) / self.domain.dy**2.0

        # d2udx2
        self.laplacian_u += (
            self.u[1:-1, 2:] - 2.0 * self.u[1:-1, 1:-1] + self.u[1:-1, :-2]
        ) / self.domain.dx**2.0

        self.laplacian_u *= self.dt * self.nu

    """ Boundary conditions. """

    def periodic_x(self) -> None:
        self.u[:, 0] = self.u[:, -2]
        self.u[:, -1] = self.u[:, 1]

    def periodic_y(self) -> None:
        self.u[0, :] = self.u[-2, :]
        self.u[-1, :] = self.u[1, :]

    def _laplacian_boundary_conditions(self) -> None:
        """
        Enforce Laplacian boundary conditions in the following order: West -> East -> South -> North.
        :return: None
        """

        lbc_funcs = []

        # if self.domain.boundary_conditions.keys()[0] == "Periodic":
        lbc_funcs.append(self.periodic_x)
        lbc_funcs.append(self.periodic_y)

        for bc, bv in self.domain.boundary_conditions.items():
            if bc == "Channel" or bc == "Basin":

                lbc_funcs.pop()

                for idx, i in enumerate([1, 3]):  # 0, 1; 1, 3
                    bi = (2, -2)[idx]

                    if len(bv[i]) == 3:

                        if i == 1:
                            # South
                            def slip_y(tmp_idx=-idx, tmp_bi=bi):
                                # idx = 0, bi = 2
                                self.u[tmp_idx, 1:-1] = (
                                        2.0 * self.u[1, 1:-1] - self.u[tmp_bi, 1:-1]
                                )
                            lbc_funcs.append(slip_y)

                        elif i == 3:
                            # North
                            def slip_y(tmp_idx=-idx, tmp_bi=bi):
                                # idx = 1, bi = -2
                                self.u[tmp_idx, 1:-1] = (
                                        2.0 * self._adv_y[0, ..., tmp_idx, :] - self.u[tmp_bi, 1:-1]
                                )
                            lbc_funcs.append(slip_y)

                    else: # len == 1

                        def dirichlet_y(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                            self.u[tmp_idx, :] = 2.0 * bv[tmp_i] - self.u[tmp_bi, :]
                        lbc_funcs.append(dirichlet_y)

            if bc == "Basin":

                lbc_funcs.pop(0)

                for idx, i in enumerate([0, 2]):  # 0, 1; 0, 2
                    bi = (2, -2)[idx]

                    if len(bv[i]) == 3:

                        if i == 0:
                            # West
                            def slip_x(tmp_idx=-idx, tmp_bi=bi):
                                self.u[1:-1, tmp_idx] = (
                                        2.0 * self.u[1:-1, 1] - self.u[1:-1, tmp_bi]
                                )
                            lbc_funcs.append(slip_x)

                        elif i == 2:
                            # East
                            def slip_x(tmp_idx=-idx, tmp_bi=bi):
                                self.u[1:-1, tmp_idx] = (
                                        2.0 * self._adv_x[0, ..., tmp_idx] - self.u[1:-1, tmp_bi]
                                )
                            lbc_funcs.append(slip_x)

                    else: # len == 1

                        def dirichlet_x(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                            self.u[:, tmp_idx] = 2.0 * bv[tmp_i] - self.u[:, tmp_bi]
                        lbc_funcs.append(dirichlet_x)
                # elif self.domain.boundary_conditions[boundaries[i]][0] in ("Neumann", "Inflow", "Outflow"):
                # def neumann_x(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                #     self.u[:, tmp_idx] = (
                #         self.u[:, tmp_bi]
                #         + (tmp_idx - tmp_bi)
                #         * self.domain.dy
                #         * self.domain.boundary_conditions[boundaries[tmp_i]][1]
                #     )
                # lbc_funcs.append(neumann_x)
            # else:
            #     if list(self.domain.boundary_conditions)[0] != "Periodic":
            #         raise ValueError(" BCs not recognised in ADR object. ")
                # def neumann_y(tmp_idx=-idx, tmp_i=i, tmp_bi=bi):
                #     self.u[tmp_idx, :] = (
                #         self.u[tmp_bi, :]
                #         + (tmp_idx - tmp_bi)
                #         * self.domain.dx
                #         * self.domain.boundary_conditions[boundaries[tmp_i]][1]
                #     )
                # lbc_funcs.append(neumann_y)
        self.laplacian_bcs = lambda: [lbc() for lbc in lbc_funcs]

    """ End of boundary conditions. """

    def update_maximum_principle(self) -> None:
        # Compute source Laplacian
        self.u[1:-1, 1:-1] = self.solutions[0]
        self.laplacian_bcs()
        self._second_order_deriv()

        self._M_x += self.laplacian_u
        self._m_x += self.laplacian_u
        self._M_y += self.laplacian_u
        self._m_y += self.laplacian_u

    def diffusion_dt(self) -> None:
        self.dt = min(self.dt, self.dt_d)

    def _corrector(self):
        # Operator splitting
        self.solutions[0] += self.laplacian_u

    def get_pre_processes(self):
        return [
            partial(self.update_dt, diff=self.diffusion_dt),
            self._boundary_conditions,
            self._laplacian_boundary_conditions,
        ]

    def get_solve_sequence(self):
        return [
            partial(self.update_dt, diff=self.diffusion_dt),
            self.predictor,
            self.numerical_fluxes,
            self.update_maximum_principle,
            self.second_order_extrapolation,
            self.numerical_fluxes,
            self.corrector,
        ]
