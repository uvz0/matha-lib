# src/__init__.py
# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------
# Import all functions from core modules and expose them

from .algebra import *
from .calculas import *
from .geometry import *
from .linalg import *
from .numTheory import *
from .stats import *
from .trigonometry import *
from .regression import *
from .machine_learning import *
from .vector import *
from .special_functions import *
from .graph import * # Assuming this module exists for plotting
from .complex import *

# Define __all__ for explicit export control (good practice)
__all__ = [
    # List functions/classes you guys want to export here, 
    # but using '*' imports above is often simpler for utility libraries. its efficent :)
]

# Set a version number for your package
__version__ = "1.0.0"