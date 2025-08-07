"""
Configuration of solver settings for the test cases.
"""

from typing import Any, Callable, Sequence

from pytest_cases import fixture  # type: ignore

from pequod import transport


@fixture(scope="class")
def result(setup: tuple[Sequence[Any], Callable]) -> transport:
    """
    Solver fixtures.
    :param setup: Initial conditions and args ->
                 [nx, ny, west, south, east, north,
                 (west_bc, south_bc, east_bc, north_bc),
                 (west_bv, south_bv, east_bv, north_bv),
                 (vx, vy), cfl, final_time]
    :return: Numerical solutions.
    """
    ic = setup[1]
    args = setup[0]

    solver = transport(*args[:6])
    solver.bcs(west=args[6][0], south=args[6][1], east=args[6][2], north=args[6][3])
    solver.bvs(west=args[7][0], south=args[7][1], east=args[7][2], north=args[7][3])
    solver.adv_vel = args[8]
    solver.cfl = args[9]
    solver.final_time = args[10]
    solver.set_ivp(ic, visual=0)
    solver.solve(save_freq=0, visual=-1)

    return solver
