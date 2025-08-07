"""
Base class to implement different types of PDE solvers.

Todo: 1) Save animation frames with proper indexing as names and save time as first line in data file
"""

import glob
import inspect
import os
from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from ..gulf import Gulf, eps64, np

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

    def __init__(self, nx, ny, left, bottom, right, top, index) -> None:
        """
        Main constructor for the solver.
        :return: 2D physics-based solver equipped with the ``CABARET`` advector, given grid dimensions.
        """

        super().__init__(nx, ny, left, bottom, right, top, index)
        self.m_t = 0.0
        self.m_final_time = 1.0
        self.last_step = False

        self.caller_dir = ""
        self.data_dir = os.path.join(self.caller_dir, "data_dir")

    @property
    def t(self):
        """
        Getter for the current simulation time.
        :return: Current time.
        """
        return self.m_t

    @t.setter
    def t(self, value: float):
        """
        Setter for the current simulation time.
        """
        self.m_t = value

    @property
    def final_time(self):
        """
        Getter for the simulation endtime.
        :return: Final time.
        """
        return self.m_final_time

    @final_time.setter
    def final_time(self, value: float):
        """
        Setter for the simulation endtime.
        """
        self.m_final_time = value

    @staticmethod
    def l2_norm(value: tuple[float | np.ndarray, float | np.ndarray]):
        return np.sqrt(np.square(value[0]) + np.square(value[1]))

    @classmethod
    def reset(cls):
        """
        Resets the solver to the new settings.
        """
        return cls(
            cls.nx_faces, cls.ny_faces, cls.west, cls.south, cls.east, cls.north, "std"
        )

    """ Visualisations and user outputs. """

    def update_pathname(self):
        """
        Directory to save data.
        """
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        self.caller_dir = os.path.dirname(os.path.abspath(caller_file))
        self.data_dir = os.path.join(self.caller_dir, "data_dir")

    def terminal_verbose(self, n):
        print(
            f" Time-step #{n}. Progress = "
            + "%.2f" % (100.0 * self.t / self.final_time)
            + " %"
        ) if n % 10 == 0 or self.last_step else None

    def visualise_sol_1d(
        self, title: str, var: int = 0, pause_time: float = 1.0
    ) -> None:
        """
        Plotting parameters and visualisations.
        """
        lw = 1.5
        y_bottom, y_top = (
            np.min(self.solutions[var][0, :]),
            np.max(self.solutions[var][0, :]),
        )

        # plt.clf()
        plt.plot(self.get_x_coords, self.solutions[var][0, :], "k", lw=lw)
        # plt.legend(loc='best')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\phi$")
        plt.axhline(0.0, linestyle=":", color="black")
        plt.ylim([y_bottom, y_top])
        plt.title(title)
        # plt.show()
        plt.pause(pause_time)
        plt.close("all")

    def visualise_sol(self, title: str, var: int = 0, pause_time: float = 1.0) -> None:
        """
        Set up figures and initial artists.
        :param title:
        :param var:
        :param pause_time:
        :return:
        """

        sol = self.solutions[var]
        z_min, z_max = np.min(sol), np.max(sol)

        # plt.clf()
        fig = plt.figure(figsize=(10, 5))

        # Subplot 1: imshow
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(
            sol,
            extent=(self.m_west, self.m_east, self.m_south, self.m_north),
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=z_min,
            vmax=z_max,
        )
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.set_xlim(self.m_west, self.m_east)
        ax1.set_ylim(self.m_south, self.m_north)
        fig.colorbar(
            im, ax=ax1, label=r"$\phi$"
        )  # cbar = Individual colour bar for imshow

        # Subplot 2: plot_surface with an adjustable camera angle
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(
            self.X, self.Y, sol, cmap="Blues_r", edgecolor="none"
        )  # surf =

        # Adjust the camera angle (elevation, azimuth)
        ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.set_zlabel(r"$\phi$")
        ax2.set_xlim(self.m_east, self.m_west)
        ax2.set_ylim(self.m_north, self.m_south)
        ax2.set_zlim([z_min, z_max])
        plt.suptitle(title, fontsize=16, usetex=True)

        # Show the plot
        plt.tight_layout()  # rect=[0, 0, 1, 0.95]
        plt.pause(pause_time)
        # plt.show()
        plt.close("all")

    @staticmethod
    def load_data(fname):
        with open(fname) as f:
            time = float(f.readline().strip())
        f.close()
        data = np.loadtxt(fname, skiprows=1)
        return time, data

    def save_animation(
        self, ani_name: str = "solutions.gif", del_data: bool = True
    ) -> None:
        file_list = sorted(glob.glob(os.path.join(self.data_dir, "data_*.dat")))
        n_frames = len(file_list)

        # Precompute global z-limits (optional: scan files to find min/max)
        sample_time, sample = self.load_data(file_list[0])
        z_min, z_max = sample.min(), sample.max()

        # plt.clf()
        fig = plt.figure(figsize=(10, 5))

        # Subplot 1: imshow
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(
            sample,
            extent=(self.m_west, self.m_east, self.m_south, self.m_north),
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=z_min,
            vmax=z_max,
        )
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.set_xlim(self.m_west, self.m_east)
        ax1.set_ylim(self.m_south, self.m_north)
        fig.colorbar(im, ax=ax1, label=r"$\phi$")

        # Subplot 2: plot_surface with an adjustable camera angle
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf = [
            ax2.plot_surface(self.X, self.Y, sample, cmap="Blues_r", edgecolor="none")
        ]

        # Adjust the camera angle (elevation, azimuth)
        ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.set_zlabel(r"$\phi$")
        ax2.set_xlim(self.m_east, self.m_west)
        ax2.set_ylim(self.m_north, self.m_south)
        ax2.set_zlim([z_min, z_max])
        plt.suptitle(
            str(self.m_nx)
            + r"$\times$"
            + str(self.m_ny)
            + " Grid. Time = "
            + "%.2f" % 0.0
            + " s",
            fontsize=16,
            usetex=True,
        )

        # Show the plot
        plt.tight_layout()

        def update(frame_idx):
            # global surf
            fname = file_list[frame_idx]
            t_val, sol = self.load_data(fname)
            s_min, s_max = sol.min(), sol.max()

            # Update 2D image
            im.set_data(sol)
            im.set_clim(s_min, s_max)

            # Replace 3D surface
            surf[0].remove()
            surf[0] = ax2.plot_surface(
                self.X, self.Y, sol, cmap="Blues_r", edgecolor="none"
            )
            ax2.set_zlim([s_min, s_max])
            ax2.view_init(elev=30, azim=45)

            # Update title to show time
            # t_str = os.path.basename(fname).split("_")[1].split(".dat")[0]
            # t_val = float(t_str)
            fig.suptitle(
                str(self.m_nx)
                + r"$\times$"
                + str(self.m_ny)
                + " Grid. Time = "
                + "%.2f" % t_val
                + " s",
                fontsize=16,
                usetex=True,
            )

            return im, surf[0]

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
            raise NotImplementedError(" Unsupported animation format.")
        plt.close(fig)

        # Clean up the data files after animation creation
        if del_data:
            for filename in file_list:
                os.remove(filename)

    """ Solver routines. """

    def set_ivp(self, ic: Callable = None, var: int = 0, visual: int = 0) -> None:
        """
        Setter for the Initial Value Problem (IVP).
        :param ic: Initial conditions.
        :param var: Variable index.
        :param visual: Plot switch.
        :return: None
        """

        # Initial datum/conditions defined
        if ic is not None:
            for i in np.arange(self.m_nx):
                for j in np.arange(self.m_ny):
                    self.m_cons[var][j][i] = ic(
                        self.get_x_coords[i], self.get_y_coords[j]
                    )

            for i in np.arange(self.m_nx_adv):
                for j in np.arange(self.m_ny):
                    self.m_adv_x[0][var][j][i] = ic(
                        self.m_x_coords_adv[i], self.m_y_coords[j]
                    )

            for i in np.arange(self.m_nx):
                for j in np.arange(self.m_ny_adv):
                    self.m_adv_y[0][var][j][i] = ic(
                        self.m_x_coords[i], self.m_y_coords_adv[j]
                    )

        if visual == 1:
            self.visualise_sol_1d(r"Initial datum", var=var)
        elif visual == 2:
            self.visualise_sol(r"Initial datum", var=var)
        else:
            pass

    @abstractmethod
    def get_pre_processes(self):
        pass

    @abstractmethod
    def get_solve_sequence(self):
        pass

    """ Main solver time-marching loop. """

    def solve(self, save_freq: int = 0, visual: int = 0) -> None:
        for pre_solve in self.get_pre_processes():
            pre_solve()

        title_str = (
            str(self.m_nx)
            + r"$\times$"
            + str(self.m_ny)
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
                fname = os.path.join(
                    self.data_dir, f"data_{int(n / save_freq):05d}.dat"
                )  # {self.t:07.4f}
                # Text-format 2D array
                np.savetxt(fname, self.solutions[0], header=f"{self.t}", comments="")
                # for faster binary use: np.save(fname.replace('.dat','.npy'), sol)

            for func in self.get_solve_sequence():
                func()

            self.t += self.dt

            # Replot
            title_str = (
                str(self.m_nx)
                + r"$\times$"
                + str(self.m_ny)
                + " Grid. Time = "
                + "%.2f" % self.t
                + " s"
            )
            visual_func(n)

            if save_freq and self.last_step:
                fname = os.path.join(
                    self.data_dir, f"data_{int(n / save_freq):05d}.dat"
                )  # {self.t:07.4f}
                np.savetxt(fname, self.solutions[0], header=f"{self.t}", comments="")

            n += 1

        print(" Simulation finished.")
