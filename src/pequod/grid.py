"""
Data class for a two-dimensional grid class object.

Notes: 3 modes of operation; 1) Doubly periodic (FFT), 2) Channel (FACR), 3) Basin (DCR)

Todo: 1) Number of grid points should be 2^n
      2) Mask arrays for irregular grid-points
"""

from typing import Self
from .pq_types import flt64, np, uint8, uint32  # noqa: F401


class Grid(object):
    """
    Base class to represent a 2D uniform grid object in physical space.
    """

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        west: float = 0.0,
        south: float = 0.0,
        east: float = 1.0,
        north: float = 1.0,
    ) -> None:
        """
        Main constructor for the computational Arakawa C-grid.

        Parameters
        ----------
        nx : uint
                Number of cells in the x-direction.
        ny : uint
                Number of cells in the y-direction.
        west : float
                y-coordinate of the west domain boundary.
        south : float
                y-coordinate of the south domain boundary.
        east : float
                x-coordinate of the east domain boundary.
        north : float
                x-coordinate of the north domain boundary.

        Returns
        -------
        None
                2D ``Grid`` with the given shape.
        """
        self._nx = uint32(nx)
        self._ny = uint32(ny)

        self._west = flt64(west)
        self._south = flt64(south)
        self._east = flt64(east)
        self._north = flt64(north)

        # self._i = range(1, self._nx - 1)
        # self._j = range(1, self._ny - 1)

        self.x_coords, self._dx = np.linspace(
            self._west, self._east, self._nx + 1, retstep=True, dtype=flt64
        )
        self.y_coords, self._dy = np.linspace(
            self._south, self._north, self._ny + 1, retstep=True, dtype=flt64
        )
        self.x_coords_adv = self.x_coords[:-1] + 0.5 * self._dx
        self.y_coords_adv = self.y_coords[:-1] + 0.5 * self._dy

        # self._ndofs = self._nx * self._ny
        # self._ndims = self._ndofs**2

        # Boundary value problem (BVP) - default settings are doubly periodic
        default_bc = 'Periodic'
        # self.pbcs = {'west': 'east', 'south': 'north', 'east': 'west', 'north': 'south'}
        self._boundary_conditions = {default_bc: ('None', 'None', 'None', 'None')}

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value: int):
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value: int):
        self._ny = value

    @property
    def west(self):
        return self._west

    @west.setter
    def west(self, value: float):
        self._west = value

    @property
    def south(self):
        return self._south

    @south.setter
    def south(self, value: float):
        self._south = value

    @property
    def east(self):
        return self._east

    @east.setter
    def east(self, value: float):
        self._east = value

    @property
    def north(self):
        return self._north

    @north.setter
    def north(self, value: float):
        self._north = value

    @property
    def dx(self) -> float:
        """
        Grid size in x-direction.
        :return: dx
        """
        return self._dx

    @property
    def dy(self) -> float:
        """
        Grid size in y-direction.
        :return: dy
        """
        return self._dy

    # @property
    # def x_coords(self) -> np.ndarray:
    #     return self.x_coords

    # @property
    # def y_coords(self) -> np.ndarray:
    #     return self.y_coords

    @property
    def boundary_conditions(self) -> dict:
        return self._boundary_conditions

    """ Boundary conditions. """

    # def bcs(self, bc_type: str = 'Periodic') -> Self:

    #     new_bcs = {bc_type: self.boundary_conditions.values()[0]}

    #     self.boundary_conditions = new_bcs

    # def bvs(self, west: float = None, south: float = None, east: float = None, north: float = None) -> Self:

    #     new_bcs = {self.boundary_conditions.keys()[0]: (west, south, east, north)}

    #     self.boundary_conditions = new_bcs

    @boundary_conditions.setter
    def boundary_conditions(self, value: dict[str, tuple[float | None | tuple[float, float, float]]]) -> None:
        
        # assert len(value.values()) == 4
        boundaries = ('west', 'south', 'east', 'north')
        bc = list(value)[0]
        if bc not in ["Periodic", "Channel", "Basin"]:
                raise NotImplementedError(f" {bc} boundary conditions not implemented. ")

        for i, val in enumerate(value.values()):
            # if bc[0] == "Periodic" and value[self.pbcs[i]][0] != "Periodic":
            #     raise ValueError(
            #         f" If the {i} boundary is Periodic, then the {self.pbcs[i]} boundary must also be Periodic. \n"
            #         f" Please change either conditions. "
            #     )
            if (bc != "Periodic" and val is None):
                raise ValueError(
                    f" The assigned boundary condition is {bc} but data given contains {None}. \n"
                    f" Please provide boundary data for {boundaries[i]}. "
                )

            if bc == "Periodic" and val is not None:
                raise UserWarning(
                    f" The {boundaries[i]} boundary is Periodic. Boundary data assigned will be overwritten. "
                )
        
        self._boundary_conditions = value
