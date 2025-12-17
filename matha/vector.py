# vector.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import numpy as np
import math

# Type Hinting for Vector: Using np.ndarray for NumPy compatibility and efficiency
Vector = np.ndarray 

# --- 1. BASIC ARITHMETIC ---

def add(v1: Vector, v2: Vector) -> Vector:
    """
    Performs vector addition (v1 + v2).
    
    Args:
        v1: The first vector.
        v2: The second vector.
        
    Returns:
        The resulting vector.
        
    Raises:
        ValueError: If vectors have different dimensions.
    """
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same dimension for addition.")
    return v1 + v2

def subtract(v1: Vector, v2: Vector) -> Vector:
    """
    Performs vector subtraction (v1 - v2).
    
    Args:
        v1: The first vector.
        v2: The vector to subtract.
        
    Returns:
        The resulting vector.
        
    Raises:
        ValueError: If vectors have different dimensions.
    """
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same dimension for subtraction.")
    return v1 - v2

def scalar_multiply(v: Vector, scalar: float) -> Vector:
    """
    Multiplies a vector v by a scalar value.
    
    Args:
        v: The input vector.
        scalar: The scalar value.
        
    Returns:
        The scaled vector.
    """
    return v * scalar

# --- 2. VECTOR PRODUCTS ---

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Calculates the dot product (scalar product) of two vectors (v1 · v2).
    The dot product is defined as: v1 · v2 = |v1| |v2| cos(θ). 
    
    Args:
        v1: The first vector.
        v2: The second vector.
        
    Returns:
        The scalar dot product.
        
    Raises:
        ValueError: If vectors have different dimensions.
    """
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same dimension for dot product.")
    return np.dot(v1, v2)

def cross_product(v1: Vector, v2: Vector) -> Vector:
    """
    Calculates the cross product (vector product) of two 3D vectors (v1 x v2).
    The result is a vector perpendicular to both input vectors.
    
    Args:
        v1: The first vector.
        v2: The second vector.
        
    Returns:
        The resulting 3D vector.
        
    Raises:
        ValueError: If vectors are not 3-dimensional.
    """
    if v1.shape != (3,) or v2.shape != (3,):
        raise ValueError("Cross product is only defined for 3-dimensional vectors (shape (3,)).")
    return np.cross(v1, v2)

# --- 3. PROPERTIES AND UTILITIES ---

def magnitude(v: Vector) -> float:
    """
    Calculates the Euclidean magnitude (length or L2 norm) of a vector |v|.
    
    Args:
        v: The input vector.
        
    Returns:
        The scalar magnitude.
    """
    return np.linalg.norm(v)

def normalize(v: Vector) -> Vector:
    """
    Converts a vector into a unit vector (magnitude of 1) in the same direction.
    
    Args:
        v: The input vector.
        
    Returns:
        The unit vector.
        
    Raises:
        ValueError: If the input vector has zero magnitude.
    """
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / mag

def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calculates the angle (in radians) between two vectors.
    
    Uses the dot product formula: θ = arccos( (v1 · v2) / (|v1| |v2|) ).
    
    Args:
        v1: The first vector.
        v2: The second vector.
        
    Returns:
        The angle in radians (0 to pi).
    """
    v1_mag = magnitude(v1)
    v2_mag = magnitude(v2)
    
    if v1_mag == 0 or v2_mag == 0:
        return 0.0 # Angle is conventionally zero if a vector is zero
    
    # Calculate the cosine of the angle
    cosine_angle = dot_product(v1, v2) / (v1_mag * v2_mag)
    
    # Clip the value to the domain [-1, 1] to prevent floating point errors
    # causing arccos to raise an error
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return math.acos(cosine_angle)