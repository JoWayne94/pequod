# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np

from pequod import transport


def ic1(x, y):
    center = [0.5, 0.5]
    std = 0.15
    return np.exp(
        (-1.0 / (2.0 * std**2)) * ((x - center[0]) ** 2 + (y - center[1]) ** 2)
    )


def gaussian_1d(x, y):
    center = [0.5, 0.0]
    std = 0.15
    return (
        np.exp((-1.0 / (2.0 * std**2)) * ((x - center[0]) ** 2 + (y - center[1]) ** 2))
        if y == 0.0
        else 0.0
    )


def top_hat_1d(x, y):
    return np.where((0.25 <= x) & (x <= 0.75) & (y == 0.0), 1.0, 0.0)


def visualise_sol_1d(x_coords, solutions, exact, title: str) -> None:
    """
    Plotting parameters and visualisations.
    """
    lw = 1.0
    y_bottom = -0.1
    y_top = 1.1

    plt.cla()
    plt.plot(x_coords, solutions, "k", lw=lw, label="Num.")
    plt.plot(x_coords, exact, "b--", lw=lw + 1, label="Exact")
    plt.legend(loc="best")
    plt.ylabel(r"$u$")
    plt.axhline(0.0, linestyle=":", color="black")
    plt.ylim([y_bottom, y_top])
    plt.title(title)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    nx = 128
    ny = 2
    x_left = 0.0
    x_right = 1.0
    y_bottom = -1.0
    y_top = 1.0

    solver = transport(nx, ny, x_left, y_bottom, x_right, y_top)
    solver.set_ivp(top_hat_1d, visual=True)
    solver.bcs(south="Dirichlet", north="Dirichlet")
    solver.bvs(south=0.0, north=0.0)

    solver.adv_vel = (1.0, 0.0)
    solver.solve(save_freq=0)
    visualise_sol_1d(
        solver.get_x_coords,
        solver.solutions[0][0, :],
        top_hat_1d(solver.get_x_coords, 0.0),
        "Numerical vs Analytical Solutions",
    )
    # solver.save_animation(del_data=False)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
