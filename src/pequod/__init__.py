__author__ = "Jo Wayne Tan"
__email__ = "jo.tan17@imperial.ac.uk"
__version__ = "1.0.0"

# Main package APIs
from .solvers.burgers import Burgers as burgers  # noqa: F401
from .solvers.transport import ScalarTransport as transport  # noqa: F401
