"""
Scalar transport equation solver class in 2D.

partial u / partial t + div F = 0
"""

from .solver import Grid, Callable, Solver, eps64, np


class ScalarTransport(Solver):
    """
    Linear advection solver on a 2D ``Grid``; discretised using the ``CABARET``.
    """

    def __init__(self, domain: Grid) -> None:
        """
        Main constructor for the ``ScalarTransport`` class.
        :return: 2D scalar transport solver object.
        """

        super().__init__(domain)
        self._cfl = 0.5
        self.max_wave_speed = 1.0
        self.dt_const = self.cfl * min(self.domain.dx, self.domain.dy)
        self.adv_vel = (1.0, 0.0)

    @property
    def cfl(self):
        """
        Getter for the Courant-Friedrichs-Lewy number.
        :return: CFL number.
        """
        return self._cfl

    @cfl.setter
    def cfl(self, value: float):
        """
        Setter for the CFL number.
        """
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("cfl must be finite and in the interval (0, 1].")
        self._cfl = value
        self.dt_const = value * min(self.domain.dx, self.domain.dy)

    @property
    def adv_vel(self):
        """
        Getter for the advective velocity.
        :return: Advective velocity.
        """
        return self._adv_vel

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
        Setter for the advective velocity field: [u, v].
        :param value: Prescribed advective velocity.
        :return: None
        """
        if type(value) is tuple and type(value[0]) is np.ndarray:
            self._adv_vel = value

        else:

            if type(value) is tuple and type(value[0]) is float:
                self.max_wave_speed = self.l2_norm(value)
                vx = value[0] * np.ones((self.domain.ny, self.nx_faces))
                vy = value[1] * np.ones((self.ny_faces, self.domain.nx))

            else:
                self.max_wave_speed = np.max(self.l2_norm((value(self.X, self.Y))))
                vx, _ = value(self.X_adv_x, self.Y_adv_x)
                _, vy = value(self.X_adv_y, self.Y_adv_y)

            self._adv_vel = (
                np.expand_dims(np.asarray(vx, dtype=np.float64), 0),
                np.expand_dims(np.asarray(vy, dtype=np.float64), 0),
            )

    def update_dt(self, adv: Callable = lambda: None, diff: Callable = lambda: None) -> None:
        """
        Compute the next time-step size, dt^{n + 1}.
        :param adv:  Update maximum wave speed in the system.
        :param diff: Handles diffusion time-step restriction.
        :return: None
        """
        adv()
        if not np.isfinite(self.max_wave_speed) or self.max_wave_speed < 0.0:
            raise FloatingPointError(
                f"Invalid maximum wave speed: {self.max_wave_speed!r}"
            )
        self.dt = (
            np.inf
            if self.max_wave_speed <= eps64
            else self.dt_const / self.max_wave_speed
        )
        diff()

        if self.t + self.dt >= self.final_time:
            self.dt = self.final_time - self.t
            self.last_step = True

    def physical_flux(self):
        """
        Flux function for the linear transport equation.
        :return: F = c(x, y; t) * u
        """
        return self._adv_x[0] * self._adv_vel[0], self._adv_y[0] * self._adv_vel[1]

    def get_pre_processes(self):
        return [self.update_dt, self._boundary_conditions]

    def get_solve_sequence(self):
        return [
            self.update_dt,
            self.predictor,
            self.second_order_extrapolation,
            self.numerical_fluxes,
            self.corrector,
        ]
