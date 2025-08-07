"""
Numerical tests for solvers.
"""

from itertools import combinations
from typing import Any, Callable, Sequence

import numpy as np
from pytest_cases import fixture, parametrize_with_cases  # type: ignore

from pequod import transport

from .transport_cases import ConvergenceCases, OneDimCases, TwoDimCases


def l2_err_measure(num: np.ndarray, exact: np.ndarray, h: float):
    return np.sqrt(np.sum(h * np.square(num - exact))) / np.sqrt(
        np.sum(h * np.square(exact))
    )


class TestImplementation:
    """
    Test ``transport`` against exact solutions.
    """

    @fixture(scope="class")
    @parametrize_with_cases(
        "setup", cases=[OneDimCases, TwoDimCases.GaussianBlob], scope="class"
    )
    def setup(self, setup) -> tuple:
        return setup

    def test_correctness(
        self, result: transport, setup: tuple[Sequence[Any], Callable]
    ):
        """
        Test errors of numerical against analytical solutions.
        """
        if result.ny == 1:
            exact = setup[1](result.get_x_coords, result.get_y_coords)
            eps = 1.0e-4
        else:
            exact = setup[1](result.X, result.Y)
            eps = 5.0e-3
        assert l2_err_measure(result.solutions[0], exact, result.dx) < eps


class TestConvergence:
    errors = []
    h_list = []

    @fixture(scope="class")
    @parametrize_with_cases("setup", cases=ConvergenceCases, scope="class")
    def setup(self, setup) -> tuple:
        return setup

    def test_correctness(
        self, result: transport, setup: tuple[Sequence[Any], Callable]
    ):
        """
        Test errors of numerical against analytical solutions.
        """
        err = l2_err_measure(
            result.solutions[0],
            setup[1](result.get_x_coords, result.get_y_coords),
            result.dx,
        )
        self.errors.append(err)
        self.h_list.append(result.dx)
        assert err < 5.0e-3

    def test_convergence(self):
        """
        Ensure the CABARET is approximately second-order accurate.
        """
        n_values = []
        for i, j in combinations(range(len(self.errors)), 2):
            epsilon_1, epsilon_2 = self.errors[i], self.errors[j]
            delta_x_1, delta_x_2 = self.h_list[i], self.h_list[j]

            # Using the formula provided to calculate n for each pair
            n = (np.log(epsilon_1) - np.log(epsilon_2)) / (
                np.log(delta_x_1) - np.log(delta_x_2)
            )
            n_values.append(n)

        # Calculate the average n
        average_n = np.mean(n_values)

        # Output the results
        print(" Numerical rates of convergence: \n", n_values)
        print(" Average r.o.c: \n", average_n)

        assert average_n > 1.6
