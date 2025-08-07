"""
Scalar transport equation solver class in 2D.

partial u / partial t + div(F) = 0
"""

from .solver import Callable, Solver, np


class ScalarTransport(Solver):
    """
    Linear advection solver on a 2D ``Grid``; discretised using the ``CABARET``.
    """

    def __init__(self, nx, ny, left, bottom, right, top, index="std") -> None:
        """
        Main constructor for the ``ScalarTransport`` class.
        :return: 2D scalar transport solver object.
        """

        super().__init__(nx, ny, left, bottom, right, top, index)
        self.m_cfl = 0.5
        self.m_max_wave_speed = 1.0
        self.dt_const = self.cfl * min(self.m_dx, self.m_dy)

        if self.m_adv_vel is None:
            self.adv_vel = (1.0, 0.0)

    @property
    def cfl(self):
        """
        Getter for the Courant-Friedrichs-Lewy number.
        :return: CFL number.
        """
        return self.m_cfl

    @cfl.setter
    def cfl(self, value: float):
        """
        Setter for the CFL number.
        """
        self.m_cfl = value
        self.dt_const = value * min(self.m_dx, self.m_dy)

    @property
    def adv_vel(self):
        """
        Getter for the advective velocity.
        :return: Advective velocity.
        """
        return self.m_adv_vel

    @adv_vel.setter
    def adv_vel(
        self,
        value: tuple[float | np.ndarray, float | np.ndarray]
        | Callable[
            [float | np.ndarray, float | np.ndarray],
            {float | np.ndarray, float | np.ndarray},
        ],
    ) -> None:
        """
        Setter for the advective velocity field: [v_x, v_y].
        :param value: Prescribed advective velocity.
        :return: None
        """
        # if type(value[0] == np.ndarray):
        #     assert value[0].ndim == 2
        #     assert value[0].shape == (self.m_nx_adv, self.m_ny)
        # if type(value[1] == np.ndarray):
        #     assert value[1].ndim == 2
        #     assert value[1].shape == (self.m_nx, self.m_ny_adv)
        # self.m_adv_vel = (value[0], value[1])

        if type(value) is tuple:
            self.m_max_wave_speed = self.l2_norm(value)
            vx = value[0] * np.ones((self.m_nx_adv, self.m_ny))
            vy = value[1] * np.ones((self.m_nx, self.m_ny_adv))

        else:
            self.m_max_wave_speed = np.max(self.l2_norm((value(self.X, self.Y))))
            vx, _ = value(self.X_adv_x, self.Y_adv_x)
            _, vy = value(self.X_adv_y, self.Y_adv_y)

        self.m_adv_vel = (
            np.expand_dims(np.asarray(vx, dtype=np.float64).T, 0),
            np.expand_dims(np.asarray(vy, dtype=np.float64).T, 0),
        )

    def update_dt(self, other: Callable = lambda: None) -> None:
        """
        Compute the next time-step size, dt^{n + 1}.
        :return: None
        """
        self.dt = self.dt_const / self.m_max_wave_speed

        other()

        if self.t + self.dt >= self.final_time:
            self.dt = self.final_time - self.t
            self.last_step = True

    def physical_flux(self):
        """
        Flux function for the linear transport equation.
        :return: F = c(x, y; t) * u
        """
        return self.m_adv_x[0] * self.m_adv_vel[0], self.m_adv_y[0] * self.m_adv_vel[1]

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
