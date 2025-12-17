# geometry.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import math
import matplotlib.pyplot as plt

# --- BASIC OPERATIONS ---

def distance(p1: tuple, p2: tuple) -> float:
    """
    Calculates the Euclidean distance between two points (x1, y1) and (x2, y2).
    
    Args:
        p1: First point tuple (x, y).
        p2: Second point tuple (x, y).
        
    Returns:
        The distance between the points.
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def midpoint(p1: tuple, p2: tuple) -> tuple:
    """
    Calculates the midpoint between two points.
    
    Args:
        p1: First point tuple (x, y).
        p2: Second point tuple (x, y).
        
    Returns:
        A tuple (mid_x, mid_y) representing the midpoint.
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def slope(points: list) -> float:
    """
    Calculates the slope of the line segment defined by the first two points.
    
    Args:
        points: A list of at least two (x, y) tuples.
        
    Returns:
        The slope (m). Returns infinity (float('inf')) for vertical lines.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required to calculate slope.")

    x1, y1 = points[0]
    x2, y2 = points[1]

    if x2 == x1:
        return float('inf')  # vertical line -> infinite slope

    return (y2 - y1) / (x2 - x1)

# --- POLYGON OPERATIONS ---

def perimeter(points: list) -> float:
    """
    Calculates the perimeter of a polygon given as a list of ordered (x, y) points.
    
    Args:
        points: A list of (x, y) tuples representing the vertices.
        
    Returns:
        The total perimeter length.
    """
    peri = 0.0
    n = len(points)
    for i in range(n):
        # Calculate distance between current point and next point (wrapping around)
        peri += distance(points[i], points[(i + 1) % n])
    return peri

def area(points: list) -> float:
    """
    Calculates the area of a polygon using the Shoelace Formula.
    
    Args:
        points: A list of ordered (x, y) tuples representing vertices.
        
    Returns:
        The area of the polygon.
    """
    n = len(points)
    # Sum of (x_i * y_{i+1})
    sum1 = sum(points[i][0] * points[(i + 1) % n][1] for i in range(n))
    # Sum of (y_i * x_{i+1})
    sum2 = sum(points[i][1] * points[(i + 1) % n][0] for i in range(n))
    
    return abs(sum1 - sum2) / 2

def centroid(points: list) -> tuple:
    """
    Calculates the geometric center (centroid) of a polygon. 
    
    Args:
        points: A list of ordered (x, y) tuples.
        
    Returns:
        A tuple (C_x, C_y) representing the centroid coordinates.
    """
    n = len(points)
    signed_area = 0.0
    Cx = 0.0
    Cy = 0.0

    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        
        # Cross product term (Shoelace factor)
        cross_product = (x0 * y1 - x1 * y0)
        signed_area += cross_product
        Cx += (x0 + x1) * cross_product
        Cy += (y0 + y1) * cross_product

    signed_area *= 0.5
    if signed_area == 0:
        raise ValueError("Polygon has zero area (points might be collinear).")
        
    Cx /= (6.0 * signed_area)
    Cy /= (6.0 * signed_area)

    return (Cx, Cy)

# --- CIRCLE OPERATIONS ---

def circle_area(radius: float) -> float:
    """Calculates the area of a circle given its radius."""
    return math.pi * radius**2

def circle_circumference(radius: float) -> float:
    """Calculates the circumference of a circle given its radius."""
    return 2 * math.pi * radius

# --- VISUALIZATION ---

def plot_polygon(points: list):
    """
    Plots the polygon defined by a list of (x, y) points using Matplotlib.
    
    Args:
        points: A list of (x, y) tuples.
    """
    # Unzip the points into X and Y lists
    x, y = zip(*points)
    
    # Repeat the first point at the end to close the shape visually
    x += (x[0],)
    y += (y[0],)
    
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', label='Vertices')
    plt.fill(x, y, alpha=0.3, label='Area')
    
    # Optional: Plot centroid if valid
    try:
        c = centroid(list(points))
        plt.plot(c[0], c[1], 'rx', label='Centroid')
    except:
        pass

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polygon Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is square so circles look like circles
    plt.show()