"""
Base class to implement different types of PDE solvers.

Todo: 1) Save animation frames with proper indexing as names and save time as first line in data file.
      2) Class methods.
      3) Option to save data at specific simulation times.
"""

import glob
import inspect
import os
from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from ..gulf import Grid, Gulf, eps64, np

# Customise visuals here
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "lines.linewidth": 1.5,
        "font.size": 16,
        "axes.labelsize": 14,
    }
)


class Solver(ABC, Gulf):
    """
    Solver base class for various physics extensions.
    """

    def __init__(self, domain: Grid) -> None:
        """
        Main constructor of the solver.
        :return: 2D physics-based solver equipped with the ``CABARET`` advector, given grid dimensions.
        """

        super().__init__(domain)
        self._t = 0.0
        self._final_time = 1.0
        self.last_step = False

        self.caller_dir = ""
        self.data_dir = os.path.join(self.caller_dir, "data_dir")
        self.ani_sol = {"data": self.solutions[0]}

    @property
    def t(self):
        """
        Getter for the current simulation time.
        :return: Current time.
        """
        return self._t

    @t.setter
    def t(self, value: float):
        """
        Setter for the current simulation time.
        """
        self._t = value

    @property
    def final_time(self):
        """
        Getter for the simulation endtime.
        :return: Final time.
        """
        return self._final_time

    @final_time.setter
    def final_time(self, value: float):
        """
        Setter for the simulation endtime.
        """
        self._final_time = value

    @staticmethod
    def sol_plt(solution):
        """
        Getter for the solution to be plotted based on boundary conditions.
        """
        return np.pad(solution, ((1, 0), (0, 1)), mode='wrap')

    @staticmethod
    def l2_norm(value: tuple[float | np.ndarray, float | np.ndarray]):
        return np.sqrt(np.square(value[0]) + np.square(value[1]))

    """ Visualisations and user outputs. """

    def update_pathname(self) -> None:
        """
        Directory to save data.
        """
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        self.caller_dir = os.path.dirname(os.path.abspath(caller_file))
        self.data_dir = os.path.join(self.caller_dir, "data_dir")

    def terminal_verbose(self, n: int, v_freq: int = 100) -> None:
        """
        Terminal verbose mode.
        :param n:      Time-step number.
        :param v_freq: Print frequency.
        :return:       None
        """
        print(
            f" Time-step #{n}. Progress = "
            + "%.2f" % (100.0 * self.t / self.final_time)
            + " %"
        ) if n % v_freq == 0 or self.last_step else None

    def visualise_sol_1d(
        self, title: str, var: int = 0, pause_time: float = 1.0
    ) -> None:
        """
        Plotting parameters and visualisations for 1D simulations.
        """
        lw = 1.5
        y_bottom, y_top = (
            np.min(self.solutions[var][0, :]),
            np.max(self.solutions[var][0, :]),
        )

        # plt.clf()
        plt.plot(self.domain.x_coords, self.sol_plt[0, :], "k", lw=lw)
        # plt.legend(loc='best')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\phi$", rotation=0)
        plt.axhline(0.0, linestyle=":", color="black")
        plt.ylim([y_bottom, y_top])
        plt.title(title)
        # plt.show()
        plt.pause(pause_time)
        plt.close("all")

    def _setup_two_panel_figure(self, solution, z_lims = None, title: str = "", use_tex = True):
        """
        Set up figures and initial artists.
        """
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)

        sol = self.sol_plt(solution["data"])
        if z_lims is None:
            z_lims = (np.min(sol), np.max(sol))

        # Subplot 1: imshow
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(
            sol,
            extent=(self.domain.west, self.domain.east, self.domain.south, self.domain.north),
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=z_lims[0],
            vmax=z_lims[1],
        )
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$", rotation=0)
        ax1.set_xlim(self.domain.west, self.domain.east)
        ax1.set_ylim(self.domain.south, self.domain.north)
        fig.colorbar(
            im, ax=ax1, label=r"$\phi$"
        )  # cbar = Individual colour bar for imshow

        # Subplot 2: plot_surface with an adjustable camera angle
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax2.plot_surface(
            self.X, self.Y, sol, cmap="Blues_r", edgecolor="none"
        )

        # Adjust the camera angle (elevation, azimuth)
        ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.set_zlabel(r"$\phi$")
        ax2.set_xlim(self.domain.east, self.domain.west)
        ax2.set_ylim(self.domain.north, self.domain.south)
        ax2.set_zlim([z_lims[0], z_lims[1]])

        title_artist = fig.suptitle(title, fontsize=16, usetex=use_tex)

        return fig, {"im": im, "surf": surf, "title": title_artist}

    def visualise_sol(self, title: str, pause_time: float = 1.0) -> None:
        """
        Visualisations for 2D simulations.
        """
        # Show the plot
        fig, _ = self._setup_two_panel_figure(self.ani_sol, title=title, use_tex=True)
        plt.pause(pause_time)
        # plt.show()
        plt.close(fig)
        # plt.tight_layout()  # rect=[0, 0, 1, 0.95]

    @staticmethod
    def load_data(f_name):
        with open(f_name) as f:
            time = float(f.readline().strip())
        f.close()
        data = np.loadtxt(f_name, skiprows=1)
        return time, data

    def save_animation(
        self, ani_name: str = "solutions.gif", del_data: bool = True) -> None:
        """
        Saves animation from solution files.
        :param ani_name: Animation file name.
        :param del_data: Delete data files after animation generation.
        :return:         None
        """
        file_list = sorted(glob.glob(os.path.join(self.data_dir, "data_*.dat")))
        n_frames = len(file_list)

        # Precompute global z-limits (optional: scan files to find min/max)
        _, sample = self.load_data(file_list[0])
        z_lims = (sample.min(), sample.max())

        # plt.clf()
        title = f"{self.domain.nx}$\\times${self.domain.ny} Grid. Time = {0.0:.2f} s"
        fig, artists = self._setup_two_panel_figure({"data": sample}, z_lims, title, use_tex=True)

        def update(frame_idx):
            # global surf
            f_name = file_list[frame_idx]
            t_val, sol = self.load_data(f_name)
            s_min, s_max = sol.min(), sol.max()

            # Update 2D image
            artists["im"].set_data(sol)
            artists["im"].set_clim(s_min, s_max)

            # Replace 3D surface
            # artists["surf"][0].remove()
            # artists["surf"][0].set_data(sol)
            # artists["surf"][0].set_clim(s_min, s_max)
            # surf[0] = ax2.plot_surface(
            #     self.X, self.Y, sol, cmap="Blues_r", edgecolor="none"
            # )
            # ax2.set_zlim([s_min, s_max])
            # ax2.view_init(elev=30, azim=45)

            # Update title to show time
            artists["title"].set_text(f"{self.domain.nx}$\\times${self.domain.ny} Grid. Time = {t_val:.2f} s")

            return artists["im"], artists["surf"], artists["title"]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=100,  # ms between frames
            blit=False,  # 3D artists don’t always support blitting
        )

        if ani_name[-3:] == "gif":
            ani.save(os.path.join(self.caller_dir, ani_name), writer="pillow")
        elif ani_name[-3:] == "mp4":
            ani.save(os.path.join(self.caller_dir, ani_name), writer="ffmpeg", fps=15)
        else:
            raise NotImplementedError(" Unsupported animation format. ")
        plt.close(fig)

        # Clean up the data files after animation creation
        if del_data:
            for filename in file_list:
                os.remove(filename)

    """ Solver routines. """

    def set_ivp(self, ic: Callable = None, var: int = 0, visual: int = 0) -> None:
        """
        Setter for the Initial Value Problem (IVP).
        :param ic:     Initial conditions.
        :param var:    Variable index.
        :param visual: No. of space dimensions plot switch.
        :return: None
        """
        # Initial datum/conditions defined
        if ic is not None:
            for j in np.arange(self.domain.ny):
                for i in np.arange(self.domain.nx):
                    self.solutions[var][j][i] = ic(
                        self.domain.x_coords[i], self.domain.y_coords[j]
                    )
                    self._adv_x[0][var][j][i] = ic(
                        self.x_coords_faces[i], self.domain.y_coords[j]
                    )
                    self._adv_y[0][var][j][i] = ic(
                        self.domain.x_coords[i], self.y_coords_faces[j]
                    )

        if visual == 1:
            self.visualise_sol_1d(r"Initial datum", var=var)
        elif visual == 2:
            self.visualise_sol(r"Initial datum")
        else:
            pass

    def save_files(self, n_steps: int, freq: int) -> None:
        """
        Save solution files into directory.
        :param n_steps: Time-step number.
        :param freq:    Data files save frequency.
        :return:        None
        """
        for k in self.ani_sol.keys():
            f_name = os.path.join(
                self.data_dir, f"{k}_{int(n_steps / freq):05d}.dat"
            )  # {self.t:07.4f}
            # Text-format 2D array
            np.savetxt(f_name, self.ani_sol[k], header=f"{self.t}", comments="")
            # for faster binary use: np.save(f_name.replace('.dat','.npy'), sol)

    """ Main solver time-marching loop. """

    @abstractmethod
    def get_pre_processes(self):
        pass

    @abstractmethod
    def get_solve_sequence(self):
        pass

    def solve(self, save_freq: int = 0, visual: int = 0) -> None:
        for pre_solve in self.get_pre_processes():
            pre_solve()

        title_str = (
            str(self.domain.nx)
            + r"$\times$"
            + str(self.domain.ny)
            + " Grid. Time = "
            + "%.2f" % self.t
            + " s"
        )
        pt = 0.01

        if visual == -1:
            visual_func = lambda x: []  # noqa: E731
        elif visual == 1:
            visual_func = lambda x: [  # noqa: E731
                self.terminal_verbose(x),
                self.visualise_sol_1d(title_str, pause_time=pt),
            ]
        elif visual == 2:
            visual_func = lambda x: [  # noqa: E731
                self.terminal_verbose(x),
                self.visualise_sol(title_str, pause_time=pt),
            ]
        else:
            visual_func = lambda x: [self.terminal_verbose(x)]  # noqa: E731

        if save_freq:
            os.makedirs(self.data_dir, exist_ok=True)

        n = 0
        while abs(self.t - self.final_time) > eps64:
            # Store every save_freq steps
            if save_freq and n % save_freq == 0:
                self.save_files(n, save_freq)

            for func in self.get_solve_sequence():
                func()

            self.t += self.dt

            # Replot
            title_str = (
                str(self.domain.nx)
                + r"$\times$"
                + str(self.domain.ny)
                + " Grid. Time = "
                + "%.2f" % self.t
                + " s"
            )
            visual_func(n)

            if save_freq and self.last_step:
                # The state at the end of the final step is a distinct frame
                # from any frame saved at the start of this same step.
                next_frame_step = (n // save_freq + 1) * save_freq
                self.save_files(next_frame_step, save_freq)

            n += 1

        print(" Simulation completed. ")
