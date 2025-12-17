# linalg.py (Formerly matrix.py)

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import numpy as np
from numpy.linalg import det, inv, solve, norm

# --- 1. CORE MATRIX OPERATIONS ---

def transpose(A: np.ndarray) -> np.ndarray:
    """
    Calculates the transpose of a matrix A (A^T).
    The rows become the columns and the columns become the rows. 

    Args:
        A: The input matrix (np.ndarray).

    Returns:
        The transpose of A.
    """
    if A.ndim not in [1, 2]:
        raise ValueError("Input must be a vector (1D) or a matrix (2D).")
    return A.T

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Performs matrix multiplication (dot product) of A and B (A · B).
    
    Args:
        A: The first matrix (np.ndarray).
        B: The second matrix (np.ndarray).

    Returns:
        The resulting matrix product.
        
    Raises:
        ValueError: If the inner dimensions do not match (A columns != B rows).
    """
    if A.shape[-1] != B.shape[0]:
        raise ValueError(
            f"Inner dimensions must match for matrix multiplication. "
            f"A has {A.shape[-1]} columns, B has {B.shape[0]} rows."
        )
    
    return A @ B # NumPy's optimized multiplication operator


def identity(n: int) -> np.ndarray:
    """
    Creates an n x n Identity Matrix (I).
    A square matrix with ones on the main diagonal and zeros elsewhere.

    Args:
        n: The dimension of the square matrix.

    Returns:
        The n x n identity matrix.
    """
    return np.eye(n)


# --- 2. MATRIX PROPERTIES ---

def matrix_determinant(A: np.ndarray) -> float:
    """
    Calculates the determinant of a square matrix A (|A|).
    The determinant indicates matrix properties like invertibility (non-zero det).

    Args:
        A: A square matrix (np.ndarray).

    Returns:
        The scalar determinant value.

    Raises:
        ValueError: If the input is not a square matrix.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square (N x N).")

    return det(A)


def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """
    Calculates the inverse of a square matrix A (A⁻¹).
    If A is non-singular, A @ A⁻¹ = I (Identity Matrix).

    Args:
        A: A square, non-singular matrix (np.ndarray).

    Returns:
        The inverse matrix A⁻¹.

    Raises:
        ValueError: If the input is not a square matrix.
        numpy.linalg.LinAlgError: If the matrix is singular (determinant is zero).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square (N x N).")
    
    # inv() handles singularity check and calculation
    return inv(A)


# --- 3. LINEAR SYSTEM SOLVER ---

def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves a system of linear equations in the form A · x = b for x.
    This is often faster and more numerically stable than calculating inv(A) @ b.

    Args:
        A: The coefficient matrix (np.ndarray, N x N).
        b: The dependent vector/matrix (np.ndarray, N x 1 or N x M).

    Returns:
        The solution vector/matrix x.

    Raises:
        numpy.linalg.LinAlgError: If A is singular or not square.
    """
    # A must be square, b must have matching rows
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        raise ValueError("A must be square (N x N) and b must have N rows.")
        
    # Uses highly optimized routines (e.g., LU decomposition)
    return solve(A, b)