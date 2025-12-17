# trigonometry.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import math
import numpy as np
from typing import Union

# --- 0. ANGLE CONVERSION ---

def to_radians(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts an angle from degrees to radians."""
    return degrees * (math.pi / 180.0)

def to_degrees(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts an angle from radians to degrees."""
    return radians * (180.0 / math.pi)

# --- ANGLE HANDLER DECORATOR ---

def _angle_handler(func):
    """
    Decorator to automatically convert the first positional argument (x) 
    from degrees to radians if the keyword argument 'in_degrees=True' is passed.
    """
    def wrapper(x: float, in_degrees: bool = False, *args, **kwargs):
        # Convert to radians before calling the core math function if in_degrees is True
        if in_degrees:
            x = to_radians(x)
        return func(x, *args, **kwargs)
    return wrapper

# --- 1. STANDARD TRIGONOMETRIC FUNCTIONS (DECORATOR APPLIED) ---

@_angle_handler
def sin(x: float) -> float:
    """Calculates the sine of an angle (default input in radians)."""
    return math.sin(x)

@_angle_handler
def cos(x: float) -> float:
    """Calculates the cosine of an angle (default input in radians)."""
    return math.cos(x)

@_angle_handler
def tan(x: float) -> float:
    """Calculates the tangent of an angle (default input in radians)."""
    return math.tan(x)

# --- 2. RECIPROCAL TRIGONOMETRIC FUNCTIONS (DECORATOR APPLIED) ---

@_angle_handler
def cosec(x: float) -> float:
    """Calculates the cosecant (1/sin(x)) of an angle."""
    s = math.sin(x)
    if s == 0:
        raise ValueError("Cosecant is undefined (sin(x) is zero).")
    return 1.0 / s

@_angle_handler
def sec(x: float) -> float:
    """Calculates the secant (1/cos(x)) of an angle."""
    c = math.cos(x)
    if c == 0:
        raise ValueError("Secant is undefined (cos(x) is zero).")
    return 1.0 / c

@_angle_handler
def cot(x: float) -> float:
    """Calculates the cotangent (cos(x)/sin(x)) of an angle."""
    s = math.sin(x)
    if s == 0:
        raise ValueError("Cotangent is undefined (sin(x) is zero).")
    return math.cos(x) / s

# --- 3. INVERSE TRIGONOMETRIC FUNCTIONS (NO DECORATOR) ---
# Inputs are ratios [-1, 1], outputs are angles (radians)

def asin(x: float) -> float:
    """Calculates the inverse sine (arcsin) of x, returning radians."""
    return math.asin(x)

def acos(x: float) -> float:
    """Calculates the inverse cosine (arccos) of x, returning radians."""
    return math.acos(x)

def atan(x: float) -> float:
    """Calculates the inverse tangent (arctan) of x, returning radians."""
    return math.atan(x)

def atan2(y: float, x: float) -> float:
    """
    Calculates the 4-quadrant inverse tangent (arctan2) of y/x, returning radians. 
    
    """
    return math.atan2(y, x)


# --- 4. HYPERBOLIC FUNCTIONS (NO DECORATOR) ---

def sinh(x: float) -> float:
    """Calculates the hyperbolic sine (sinh) of x."""
    return math.sinh(x)

def cosh(x: float) -> float:
    """Calculates the hyperbolic cosine (cosh) of x."""
    return math.cosh(x)

def tanh(x: float) -> float:
    """Calculates the hyperbolic tangent (tanh) of x."""
    return math.tanh(x)


# --- 5. INVERSE HYPERBOLIC FUNCTIONS (NO DECORATOR) ---

def asinh(x: float) -> float:
    """Calculates the inverse hyperbolic sine (arcsinh) of x."""
    return math.asinh(x)

def acosh(x: float) -> float:
    """Calculates the inverse hyperbolic cosine (arccosh) of x (input x >= 1)."""
    return math.acosh(x)

def atanh(x: float) -> float:
    """Calculates the inverse hyperbolic tangent (arctanh) of x (input -1 < x < 1)."""
    return math.atanh(x)

# --- 6. TRIGONOMETRIC IDENTITIES / RELATIONS (DECORATOR LOGIC INTEGRATED) ---

def law_of_cosines(a: float, b: float, angle_c: float, in_degrees: bool = False) -> float:
    """
    Calculates the length of side c using the Law of Cosines: 
    c² = a² + b² - 2ab * cos(C). 
    
    Args:
        a: Length of side a.
        b: Length of side b.
        angle_c: The angle C opposite to side c.
        in_degrees: Set True if angle_c is in degrees.
        
    Returns:
        The length of side c.
    """
    # Use the cos function which now correctly handles the angle conversion
    c_sq = a**2 + b**2 - 2 * a * b * cos(angle_c, in_degrees=in_degrees)
    
    if c_sq < 0:
        return 0.0
        
    return math.sqrt(c_sq)

def pythagorean_identity(theta: float, in_degrees: bool = False) -> float:
    """
    Verifies the fundamental Pythagorean identity: sin²(θ) + cos²(θ).
    
    Args:
        theta: The angle θ.
        in_degrees: Set True if theta is in degrees.
        
    Returns:
        The result (should be 1.0, modulo floating-point error).
    """
    # Use the sin/cos functions which now correctly handle the angle conversion
    return sin(theta, in_degrees=in_degrees)**2 + cos(theta, in_degrees=in_degrees)**2

# :)
# --- 7. ALIASES (For user convenience) ---

arcsine = asin
arccosine = acos
arctangent = atan
arctangent2 = atan2

# Hyperbolic aliases
hsin = sinh
hcos = cosh
htan = tanh
inverse_hyperbolic_sin = asinh
inverse_hyperbolic_cos = acosh
inverse_hyperbolic_tan = atanh

"""
    In any trigonometric function, if the value is in degrees,
    in your function call, set the keyword argument `in_degrees=True` :)
"""