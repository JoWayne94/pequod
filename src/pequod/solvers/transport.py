"""
Two-dimensional scalar transport equation solver class

partial u / partial t + div(F) = 0
"""

from .solver import *


class ScalarTransport(Solver):
    """
    Scalar advection solver; child class of a 2D ``Grid`` discretised using the ``CABARET``.
    """

    def __init__(self, nx, ny, bottom, top, left, right, index="std") -> None:
        """
        Main constructor for the ``ScalarTransport`` class.
        :return: 2D scalar transport solver class object.
        """

        super().__init__(nx, ny, bottom, top, left, right, index)
        self.m_max_wave_speed = 1.0

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
        value: tuple[float, float]
        | Callable[
            [float | np.ndarray, float | np.ndarray],
            {float | np.ndarray, float | np.ndarray},
        ],
    ):
        """
        Setter for the advective velocity: [v_x, v_y].
        """
        if type(value) is tuple:
            self.m_max_wave_speed = self.l2_norm(value)
            vx = value[0] * np.ones((self.m_nx_adv, self.m_ny))
            vy = value[1] * np.ones((self.m_nx, self.m_ny_adv))

        else:
            self.m_max_wave_speed = np.max(self.l2_norm((value(self.X, self.Y))))
            vx, _ = value(self.X_adv_x, self.Y_adv_x)
            _, vy = value(self.X_adv_y, self.Y_adv_y)

        self.set_advection_velocity((vx, vy))

    # def flux_def(self, u):
    #     """
    #     Flux function for the transport equation.
    #
    #     :param u:  float or np.ndarray. Advected scalar quantity
    #
    #     :return: F = c(x, y; t) * u
    #     """
    #     return self.m_adv_vel * u

    def solve(self, save_freq: int = 0) -> None:
        if self.m_adv_vel is None:
            self.set_advection_velocity((1.0, 0.0))

        self.dt = self.cfl * min(self.m_dx, self.m_dy) / self.m_max_wave_speed

        # Directory to save data
        data_dir = "data_dir"
        if save_freq:
            os.makedirs(data_dir, exist_ok=True)

        n = 0
        last_step = False

        while abs(self.t - self.final_time) > 1.0e-8:
            if self.t + self.dt >= self.final_time:
                self.dt = self.final_time - self.t
                last_step = True

            # Store every save_freq steps
            if save_freq and n % save_freq == 0:
                fname = os.path.join(data_dir, f"ps_{self.t:07.4f}.dat")
                # text-format 2D array
                np.savetxt(fname, self.solutions[0])
                # for faster binary use: np.save(fname.replace('.dat','.npy'), sol)

            self.predictor_corrector()

            self.second_order_extrapolation()

            self.numerical_flux()

            self.enforce_boundary_conditions()

            self.predictor_corrector()

            self.t += self.dt

            # Replot
            # self.visualise_sol(str(self.m_nx) + r'$\times$' + str(self.m_ny) + ' Grid. Time = ' + '%.2f' % self.t + ' s',
            #                    pause_time=0.1)

            if save_freq and last_step:
                fname = os.path.join(data_dir, f"ps_{self.t:07.4f}.dat")
                # text-format 2D array
                np.savetxt(fname, self.solutions[0])

            n += 1

        print(" Simulation finished.")

        # # Save current frame as an image file
        # frame_filename = os.path.join(frame_dir, f'frame_{n:03d}.png')
        # plt.savefig(frame_filename)
        # frame_files.append(frame_filename)
        #
        # # Create the GIF
        # with imageio.get_writer('ani.gif', mode='I', duration=0.002) as writer:
        #     for filename in frame_files:
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        #
        # # Optional: Clean up the frame files after GIF creation
        # for filename in frame_files:
        #     os.remove(filename)
