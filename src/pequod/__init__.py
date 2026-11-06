__author__ = "Jo Wayne Tan"
__email__ = "jo.tan17@imperial.ac.uk"
__version__ = "1.0.0"

# Main package APIs
from .solvers.adr import ADR as adr  # noqa: F401
from .solvers.burgers import Burgers as burgers  # noqa: F401
from .solvers.transport import ScalarTransport as transport  # noqa: F401
from .solvers.incns import IncNavierStokes as incns  # noqa: F401
from .solvers.qg import QuasiGeostrophic as qg  # noqa: F401
from .solvers.qg2 import TwoLayerQG as qg2  # noqa: F401
from .grid import Grid as grid  # noqa: F401
