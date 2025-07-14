# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

from pequod import transport


def ic1(x, y):
    center = [0.5, 0.5]
    std = 0.15
    return np.exp(
        (-1.0 / (2.0 * std**2)) * ((x - center[0]) ** 2 + (y - center[1]) ** 2)
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    nx = 64
    ny = nx
    x_left = 0.0
    x_right = 1.0
    y_bottom = 0.0
    y_top = 1.0

    solver = transport(nx, ny, x_left, y_bottom, x_right, y_top)
    solver.set_ivp(ic1, visual=True)

    solver.adv_vel = (1.0, 1.0)
    solver.solve(save_freq=10)
    solver.save_animation(del_data=False)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
