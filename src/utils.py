import matplotlib.pyplot as plt
import numpy as np


def visualise_sol_1d(x_coords, solutions, exact, title: str) -> None:
    """
    Plotting parameters and visualisations.
    """
    lw = 1.0
    ms = 3.0
    x_left, x_right = np.min(x_coords), np.max(x_coords)

    # plt.clf()
    plt.plot(x_coords, solutions, "ko", lw=lw, ms=ms, label="Num.")
    plt.plot(x_coords, exact, "b-", lw=lw + 1, label="Exact")
    plt.legend(loc="best")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\phi$")
    plt.axhline(0.0, linestyle=":", color="black")
    plt.xlim([x_left, x_right])
    plt.title(title)
    plt.show()
