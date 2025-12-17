# stats.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import numpy as np
import math
from collections import Counter
from typing import Union, List, Tuple
from numpy.typing import ArrayLike

# Type alias for clarity
Data = Union[List[float], np.ndarray]

# --- 1. DESCRIPTIVE STATISTICS (Measures of Center) ---

def mean(data: Data) -> float:
    """Calculates the arithmetic mean (average) of the data."""
    if not data:
        return np.nan
    return sum(data) / len(data)

def median(data: Data) -> float:
    """
    Calculates the median (middle value) of the data.
    
    Args:
        data: A list or array of numerical data.
        
    Returns:
        The median value.
    """
    data_list = sorted(list(data))
    n = len(data_list)
    mid = n // 2
    
    if n % 2 == 0:
        # Even number of elements: average of the two middle values
        return (data_list[mid - 1] + data_list[mid]) / 2
    else:
        # Odd number of elements: the single middle value
        return float(data_list[mid])
    
def mode(data: Data) -> List[float]:
    """
    Calculates the mode(s) (most frequently occurring value(s)) of the data.
    
    Args:
        data: A list or array of numerical data.
        
    Returns:
        A list of mode(s) (can be multimodal).
    """
    if not data:
        return []
        
    freq = Counter(data)
    max_count = max(freq.values())
    
    return [val for val, count in freq.items() if count == max_count]

# --- 2. MEASURES OF SPREAD ---

def data_range(data: Data) -> float:
    """Calculates the range (max - min) of the data."""
    if not data:
        return np.nan
    return max(data) - min(data)

def variance(data: Data, sample: bool = True) -> float:
    """
    Calculates the variance of the data. 
    
    Args:
        data: The input data.
        sample: If True (default), calculates sample variance (divides by n-1). 
                If False, calculates population variance (divides by n).
    """
    n = len(data)
    if n < 1 or (sample and n < 2):
        return np.nan
        
    mu = mean(data)
    degrees_of_freedom = (n - 1) if sample else n
    
    return sum((x - mu) ** 2 for x in data) / degrees_of_freedom

def std_dev(data: Data, sample: bool = True) -> float:
    """Calculates the standard deviation (square root of the variance)."""
    return variance(data, sample) ** 0.5

def z_score(x: float, data: Data, sample: bool = True) -> float:
    """
    Calculates the Z-score (standard score) for a value x relative to the data.
    Z = (x - mean) / std_dev
    
    Args:
        x: The value to calculate the Z-score for.
        data: The dataset.
        sample: Use sample standard deviation if True.
    """
    mu = mean(data)
    sigma = std_dev(data, sample=sample)
    
    if sigma == 0:
        return np.nan
    return (x - mu) / sigma

# --- 3. QUANTILES & OUTLIERS ---

def percentile(data: Data, p: float) -> float:
    """
    Calculates the p-th percentile of the data using linear interpolation 
    between the nearest ranks (Type 7 method used by numpy and R).
    
    Args:
        data: The input data.
        p: The percentile rank (0 to 100).
        
    Returns:
        The value at the specified percentile.
    """
    data_list = sorted(list(data))
    n = len(data_list)
    
    # Calculate the index position (k)
    k = (n - 1) * (p / 100)
    
    # Floor and Ceiling indices
    f = int(np.floor(k))
    c = int(np.ceil(k))
    
    if f == c:
        return data_list[int(k)]
        
    # Linear interpolation: data[f] * (c - k) + data[c] * (k - f)
    # This is equivalent to data[f] + (data[c] - data[f]) * (k - f)
    return data_list[f] * (c - k) + data_list[c] * (k - f)


def quartiles(data: Data) -> Tuple[float, float, float]:
    """
    Calculates the 1st, 2nd (Median), and 3rd quartiles (Q1, Q2, Q3).
    
    Returns:
        A tuple (Q1, Q2, Q3).
    """
    return (
        percentile(data, 25), 
        percentile(data, 50), 
        percentile(data, 75)
    )

def iqr(data: Data) -> float:
    """
    Calculates the Interquartile Range (IQR = Q3 - Q1).
    A robust measure of statistical dispersion.
    """
    Q1, _, Q3 = quartiles(data)
    return Q3 - Q1

# --- 4. RELATIONAL STATISTICS ---

def covariance(x: Data, y: Data) -> float:
    """
    Calculates the sample covariance between two datasets (x and y).
    
    Args:
        x: The first dataset.
        y: The second dataset.
        
    Returns:
        The covariance value.
        
    Raises:
        ValueError: If datasets have different lengths.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
        
    n = len(x)
    if n < 2:
        return np.nan
        
    mu_x = mean(x)
    mu_y = mean(y)
    
    # Sample covariance: sum((xi - mux) * (yi - muy)) / (n - 1)
    return sum((a - mu_x) * (b - mu_y) for a, b in zip(x, y)) / (n - 1)

def correlation(x: Data, y: Data) -> float:
    """
    Calculates the Pearson correlation coefficient (r) between two datasets.
    r = Covariance(x, y) / (StdDev(x) * StdDev(y)) 

[Image of Correlation Coefficient formula]

    
    Returns:
        The correlation coefficient (-1.0 to 1.0).
    """
    std_x = std_dev(x)
    std_y = std_dev(y)
    
    if std_x == 0 or std_y == 0:
        return np.nan # Undefined if either dataset has zero variance

    return covariance(x, y) / (std_x * std_y)

def corr_matrix(data: ArrayLike) -> np.ndarray:
    """
    Calculates the correlation matrix for a multi-dimensional dataset.
    
    Args:
        data: A 2D array or list of lists (features as columns, observations as rows).
        
    Returns:
        The symmetric correlation matrix.
    """
    # Assuming standard NumPy convention: features are columns (rowvar=False)
    return np.corrcoef(data, rowvar=False)

# --- 5. DISCRETE PROBABILITY DISTRIBUTIONS ---

def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Probability Mass Function (PMF) for the Binomial Distribution.
    P(X=k) = nCk * p^k * (1-p)^(n-k)
    
    Args:
        k: Number of successful outcomes.
        n: Number of trials.
        p: Probability of success on a single trial (0 < p < 1).
    """
    # Use math.comb (nCr)
    return math.comb(n, k) * (p**k) * ((1 - p)**(n - k))

def binomial_cdf(k: int, n: int, p: float) -> float:
    """
    Cumulative Distribution Function (CDF) for the Binomial Distribution.
    P(X <= k) = sum(P(X=i) for i in 0 to k)
    """
    if k < 0: return 0.0
    if k >= n: return 1.0
    return sum(binomial_pmf(i, n, p) for i in range(0, k + 1))

def geometric_pmf(k: int, p: float) -> float:
    """
    Probability Mass Function (PMF) for the Geometric Distribution (first success).
    P(X=k) = (1-p)^(k-1) * p
    
    Args:
        k: Number of trials until the first success (k >= 1).
        p: Probability of success (0 < p < 1).
    """
    if k < 1: return 0.0
    return (1 - p)**(k - 1) * p

def geometric_cdf(k: int, p: float) -> float:
    """
    Cumulative Distribution Function (CDF) for the Geometric Distribution.
    P(X <= k) = 1 - (1-p)^k
    """
    if k < 1: return 0.0
    return 1 - (1 - p)**k

def poisson_pmf(k: int, lam: float) -> float:
    """
    Probability Mass Function (PMF) for the Poisson Distribution.
    P(X=k) = (λ^k * e^-λ) / k! 
    
    Args:
        k: Number of occurrences.
        lam: Average rate of occurrences (λ).
    """
    if k < 0: return 0.0
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def poisson_cdf(k: int, lam: float) -> float:
    """
    Cumulative Distribution Function (CDF) for the Poisson Distribution.
    P(X <= k) = sum(P(X=i) for i in 0 to k)
    """
    if k < 0: return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

# --- 6. CONTINUOUS PROBABILITY DISTRIBUTIONS ---

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Probability Density Function (PDF) for the Normal (Gaussian) Distribution.
    
    Args:
        x: The value at which to evaluate the PDF.
        mu: The mean (μ).
        sigma: The standard deviation (σ).
    """
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive.")
        
    # PDF formula: 1/(sigma*sqrt(2pi)) * exp(-((x-mu)^2)/(2*sigma^2))
    factor = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu)**2) / (2 * sigma**2)
    return factor * np.exp(exponent)

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Cumulative Distribution Function (CDF) for the Normal Distribution.
    Uses the relationship with the Error Function (erf). 
    
    Args:
        x: The value at which to evaluate the CDF.
        mu: The mean (μ).
        sigma: The standard deviation (σ).
    """
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive.")
        
    # CDF formula: 0.5 * (1 + erf((x-mu) / (sigma*sqrt(2))))
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))