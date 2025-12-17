# complex.py

# -------------------------
# @Author : AstroJr0
# @Date : 17-12-2025
# -------------------------

import cmath
from typing import Union, Tuple

# Type Hinting for Complex Number
Complex = complex 

# --- 1. CREATION AND PROPERTIES ---

def create(real: float, imag: float) -> Complex:
    """Creates a complex number z = a + bj."""
    return complex(real, imag)

def polar(z: Complex) -> Tuple[float, float]:
    """Returns (magnitude, phase_in_radians)."""
    return cmath.polar(z)

def conjugate(z: Complex) -> Complex:
    """Returns the complex conjugate z* (a - bj)."""
    return z.conjugate()

# --- 2. BASIC ARITHMETIC ---

def add(z1: Complex, z2: Complex) -> Complex:
    return z1 + z2

def subtract(z1: Complex, z2: Complex) -> Complex:
    return z1 - z2

def multiply(z1: Complex, z2: Complex) -> Complex:
    return z1 * z2

def divide(z1: Complex, z2: Complex) -> Complex:
    if z2 == 0:
        raise ZeroDivisionError("Cannot divide by zero complex number.")
    return z1 / z2

# --- 3. COMMON MATHEMATICAL FUNCTIONS ---

def magnitude(z: Complex) -> float:
    """Returns the absolute value |z|."""
    return abs(z)

def phase(z: Complex) -> float:
    """Returns the phase angle in radians."""
    return cmath.phase(z)

def complex_sqrt(z: Complex) -> Complex:
    return cmath.sqrt(z)

def complex_log(z: Complex) -> Complex:
    return cmath.log(z)

# --- 4. COMPLEX TRIGONOMETRY ---

def c_sin(z: Complex) -> Complex:
    return cmath.sin(z)

def c_cos(z: Complex) -> Complex:
    return cmath.cos(z)

def c_tan(z: Complex) -> Complex:
    return cmath.tan(z)

# Reciprocals
def c_cosec(z: Complex) -> Complex:
    return 1.0 / cmath.sin(z)

def c_sec(z: Complex) -> Complex:
    return 1.0 / cmath.cos(z)

def c_cot(z: Complex) -> Complex:
    return 1.0 / cmath.tan(z)

# --- 5. COMPLEX HYPERBOLIC FUNCTIONS ---

def c_sinh(z: Complex) -> Complex:
    return cmath.sinh(z)

def c_cosh(z: Complex) -> Complex:
    return cmath.cosh(z)

def c_tanh(z: Complex) -> Complex:
    """Calculates the complex hyperbolic tangent of z."""
    return cmath.tanh(z)

# --- 6. INVERSE HYPERBOLIC FUNCTIONS ---

def c_asinh(z: Complex) -> Complex:
    """Calculates the inverse complex hyperbolic sine of z."""
    return cmath.asinh(z)

def c_acosh(z: Complex) -> Complex:
    """Calculates the inverse complex hyperbolic cosine of z."""
    return cmath.acosh(z)

def c_atanh(z: Complex) -> Complex:
    """Calculates the inverse complex hyperbolic tangent of z."""
    return cmath.atanh(z)

# --- 7. INVERSE TRIGONOMETRIC FUNCTIONS ---

def c_asin(z: Complex) -> Complex:
    return cmath.asin(z)

def c_acos(z: Complex) -> Complex:
    return cmath.acos(z)

def c_atan(z: Complex) -> Complex:
    return cmath.atan(z)