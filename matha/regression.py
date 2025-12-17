# regression.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import numpy as np

# --- 1. METRICS ---

def mse(y_true, y_pred) -> float:
    """
    Calculates Mean Squared Error (MSE).
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        
    Returns:
        The MSE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_pred - y_true)**2)

def mae(y_true, y_pred) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        
    Returns:
        The MAE value.
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def r_squared(y_true, y_pred) -> float:
    """
    Calculates the R-squared (coefficient of determination) score.
    Represents the proportion of variance in the dependent variable explained by the model.
    
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        
    Returns:
        The R-squared score (best is 1.0).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred)**2) # Residual Sum of Squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2) # Total Sum of Squares
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

# --- 2. SOLVERS ---

def linear_regression_analytical(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculates the optimal weights (coefficients) for Simple Linear Regression 
    using the closed-form Normal Equation.
    
    For a linear model y = mX + b:
    - m = Sum((Xi - mean(X)) * (Yi - mean(Y))) / Sum((Xi - mean(X))^2)
    - b = mean(Y) - m * mean(X)
    
    Args:
        X: The feature array (1D).
        y: The target array (1D).
        
    Returns:
        A tuple (slope, intercept) or (m, b).
        
    Raises:
        ValueError: If X has zero variance.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    X_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean)**2)

    if denominator == 0:
        raise ValueError("Cannot perform linear regression: Feature X has zero variance (all values are the same).")

    m = numerator / denominator
    b = y_mean - m * X_mean

    return m, b


def gradient_descent(x: np.ndarray, y: np.ndarray, lr: float = 0.01, iterations: int = 1000) -> tuple[float, float]: 
    """
    Finds the optimal slope (m) and intercept (b) for Simple Linear Regression 
    using the Gradient Descent optimization algorithm.
    
    
    Args:
        x: The feature array (1D).
        y: The target array (1D).
        lr: Learning rate (step size).
        iterations: Number of training steps.
        
    Returns:
        A tuple (m, b) representing the optimized slope and intercept.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    m, b = 0.0, 0.0 # Initialize weights
    n = len(x) 
    
    for _ in range(iterations): 
        y_pred = m * x + b 
        
        # Calculate gradients (partial derivatives of MSE loss function)
        dm = (-2 / n) * np.sum(x * (y - y_pred)) 
        db = (-2 / n) * np.sum(y - y_pred) 
        
        # Update weights
        m -= lr * dm 
        b -= lr * db 
        
    # FIX: Return the final optimized values after the loop completes
    return m, b

def predict(X: np.ndarray, m: float, b: float) -> np.ndarray:
    """
    Makes predictions using a simple linear regression model (y = mX + b).
    
    Args:
        X: The input feature array.
        m: The model's slope (coefficient).
        b: The model's intercept.
        
    Returns:
        The predicted values array (y_pred).
    """
    X = np.array(X, dtype=float)
    return m * X + b