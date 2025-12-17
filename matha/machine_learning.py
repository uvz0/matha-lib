# machine_learning.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import numpy as np

# --- 1. REGRESSION METRICS ---

def mse(pred, actual) -> float:
    """
    Calculates Mean Squared Error (MSE).
    Penalizes larger errors more severely than MAE.
    
    Args:
        pred: Predicted values.
        actual: Actual target values.
    """
    pred = np.array(pred)
    actual = np.array(actual)
    return np.mean((pred - actual)**2)

def rmse(pred, actual) -> float:
    """
    Calculates Root Mean Squared Error (RMSE).
    metric is in the same units as the target variable.
    """
    return np.sqrt(mse(pred, actual))

def mae(pred, actual) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    Average magnitude of errors without considering direction.
    """
    return np.mean(np.abs(np.array(pred) - np.array(actual)))

def r2_score(pred, actual) -> float:
    """
    Calculates the R-squared (coefficient of determination) score.
    Represents the proportion of variance in the dependent variable explained by the model.
    Best score is 1.0.
    """
    y_true = np.array(actual)
    y_pred = np.array(pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2) # Residual Sum of Squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # Total Sum of Squares
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

# --- 2. CLASSIFICATION METRICS ---

def accuracy_score(pred, actual) -> float:
    """
    Calculates the accuracy classification score.
    (Number of correct predictions) / (Total number of predictions).
    """
    pred = np.array(pred)
    actual = np.array(actual)
    return np.mean(pred == actual)

def binary_cross_entropy(pred, actual, epsilon=1e-15) -> float:
    """
    Calculates Binary Cross Entropy (Log Loss).
    Used for binary classification tasks. 
    
    Args:
        pred: Predicted probabilities (0 to 1).
        actual: Actual binary labels (0 or 1).
        epsilon: Small value to avoid log(0).
    """
    p = np.clip(pred, epsilon, 1 - epsilon) # Clip to avoid log(0) error
    y = np.array(actual)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def confusion_matrix_basic(pred, actual):
    """
    Computes a basic 2x2 confusion matrix for binary classification.
    Returns: (TP, TN, FP, FN)
    """
    pred = np.array(pred)
    actual = np.array(actual)
    
    tp = np.sum((pred == 1) & (actual == 1))
    tn = np.sum((pred == 0) & (actual == 0))
    fp = np.sum((pred == 1) & (actual == 0))
    fn = np.sum((pred == 0) & (actual == 1))
    
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

# --- 3. ACTIVATION FUNCTIONS ---

def sigmoid(x):
    """
    Sigmoid activation function. 
    Maps input values to a range (0, 1). Used in logistic regression. 

[Image of Sigmoid function curve]

    """
    return 1 / (1 + np.exp(-np.array(x)))

def relu(x):
    """
    Rectified Linear Unit (ReLU).
    f(x) = max(0, x). Standard activation for hidden layers in Deep Learning. 
    """
    return np.maximum(0, np.array(x))

def softmax(x):
    """
    Softmax activation function.
    Converts a vector of numbers into a vector of probabilities summing to 1.
    """
    x = np.array(x)
    # Subtract max for numerical stability (prevents overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# --- 4. PREPROCESSING UTILS ---

def train_test_split(X, y, test_size=0.2, shuffle=True):
    """
    Splits arrays or matrices into random train and test subsets.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of the dataset to include in the test split (0.0 to 1.0).
        shuffle: Whether to shuffle the data before splitting.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
        
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def min_max_scale(data):
    """
    Scales data to the range [0, 1].
    Formula: (x - min) / (max - min)
    """
    data = np.array(data)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    # Avoid division by zero
    diff = max_val - min_val
    diff[diff == 0] = 1.0 
    
    return (data - min_val) / diff