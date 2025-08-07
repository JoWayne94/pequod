"""
Scalar inviscid Burgers' equation solver class in 2D.

partial u / partial t + div(F) = 0; F = u^2 / 2
"""

from .transport import ScalarTransport, np


class Burgers(ScalarTransport):
    """
    Scalar inviscid Burgers' equation solver, inherited from ``ScalarTransport``.
    """

    def __init__(self, nx, ny, left, bottom, right, top, index="std") -> None:
        """
        Main constructor for the ``Burgers'`` class.
        :return: 2D scalar nonlinear transport solver object.
        """

        super().__init__(nx, ny, left, bottom, right, top, index)

    def physical_flux(self):
        """
        Flux function for the scalar Burgers' equation.
        :return: F = 0.5 * u^2
        """
        return 0.5 * self.m_adv_x[0] * self.m_adv_x[0], 0.5 * self.m_adv_y[
            0
        ] * self.m_adv_y[0]

    def periodic_adv_vel_x(self):
        self.m_adv_vel[0][..., :-1] = 0.5 * (
            self.m_cons + np.roll(self.m_cons, 1, axis=-1)
        )
        self.m_adv_vel[0][..., -1] = self.m_adv_vel[0][..., 0].copy()

    def periodic_adv_vel_y(self):
        self.m_adv_vel[1][..., :-1, :] = 0.5 * (
            self.m_cons + np.roll(self.m_cons, 1, axis=-2)
        )
        self.m_adv_vel[1][..., -1, :] = self.m_adv_vel[1][..., 0, :].copy()

    def internal_adv_vel_x(self):
        self.m_adv_vel[0][..., 1:-1] = 0.5 * (
            self.m_cons[..., 1:] + self.m_cons[..., :-1]
        )

    def internal_adv_vel_y(self):
        self.m_adv_vel[1][..., 1:-1, :] = 0.5 * (
            self.m_cons[..., 1:, :] + self.m_cons[..., :-1, :]
        )

    def dirichlet_adv_vel_west(self):
        self.m_adv_vel[0][..., 0] = self.m_boundary_values[0] * np.ones(self.m_ny)

    def dirichlet_adv_vel_east(self):
        self.m_adv_vel[0][..., -1] = self.m_boundary_values[2] * np.ones(self.m_ny)

    def dirichlet_adv_vel_south(self):
        self.m_adv_vel[1][..., 0, :] = self.m_boundary_values[1] * np.ones(self.m_nx)

    def dirichlet_adv_vel_north(self):
        self.m_adv_vel[1][..., -1, :] = self.m_boundary_values[3] * np.ones(self.m_nx)

    def neumann_adv_vel_west(self):
        self.m_adv_vel[0][..., 0] = self.m_cons[..., 0].copy()

    def neumann_adv_vel_east(self):
        self.m_adv_vel[0][..., -1] = self.m_cons[..., -1].copy()

    def neumann_adv_vel_south(self):
        self.m_adv_vel[1][..., 0, :] = self.m_cons[..., 0, :].copy()

    def neumann_adv_vel_north(self):
        self.m_adv_vel[1][..., -1, :] = self.m_cons[..., -1, :].copy()

    def init_vel(self):
        self.m_adv_vel = (self.m_adv_x[0], self.m_adv_y[0])

    def update_adv_vel(self) -> None:
        # self.m_max_wave_speed = max(np.max(np.abs(self.m_adv_x[0])), np.max(np.abs(self.m_adv_y[0])))
        self.m_max_wave_speed = np.max(np.abs(self.m_cons[0]))

    def get_pre_processes(self):
        tmp_func = lambda: [  # noqa: E731
            self.enforce_boundary_conditions(
                periodic_x=[self.periodic_adv_vel_x],
                periodic_y=[self.periodic_adv_vel_y],
                internal_x=[self.internal_adv_vel_x],
                internal_y=[self.internal_adv_vel_y],
                dirichlet=(
                    [self.dirichlet_adv_vel_west],
                    [self.dirichlet_adv_vel_south],
                    [self.dirichlet_adv_vel_east],
                    [self.dirichlet_adv_vel_north],
                ),
                neumann=(
                    [self.neumann_adv_vel_west],
                    [self.neumann_adv_vel_south],
                    [self.neumann_adv_vel_east],
                    [self.neumann_adv_vel_north],
                ),
            )
        ]

        return [self.init_vel, tmp_func]

    def get_solve_sequence(self):
        return [
            self.update_adv_vel,
            self.update_dt,
            self.predictor,
            self.second_order_extrapolation,
            self.numerical_fluxes,
            self.corrector,
        ]
