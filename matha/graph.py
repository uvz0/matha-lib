# graph.py (Visualizations and Plotting)

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

# --- BASIC STATISTICAL PLOTS ---

def histogram(data, bins: int = 10, title: str = "Histogram", 
            edgecolor: str = 'black', xlabel: str = "Value", ylabel: str = "Frequency"):
    """
    Plots a histogram to show the frequency distribution of a dataset.
    
    Args:
        data: A list or array of numerical data.
        bins: Number of bins (intervals) for the histogram.
        title: Chart title.
        edgecolor: Color of the bin edges.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, edgecolor=edgecolor)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def boxplot(data, title: str = "Boxplot", ylabel: str = "Values"):
    """
    Creates a boxplot to visualize statistical five-number summary:
    minimum, first quartile, median, third quartile, and maximum.
    
    Args:
        data: A list or array of numerical data.
        title: Chart title.
        ylabel: Label for the Y-axis.
    """
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, vert=True, patch_artist=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def violin_plot(data, title: str = "Violin Plot", ylabel: str = "Values"):
    """
    Creates a violin plot, which combines a boxplot with a kernel density plot.
    Useful for seeing the shape of the data distribution.
    
    Args:
        data: A list or array of numerical data.
        title: Chart title.
        ylabel: Label for the Y-axis.
    """
    plt.figure(figsize=(5, 4))
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()

# --- ADVANCED DISTRIBUTION PLOTS ---

def density_plot(data, title: str = "Density Plot", xlabel: str = "Value", ylabel: str = "Density"):
    """
    Plots the Probability Density Function (PDF) using Gaussian Kernel Density Estimation (KDE).
    This creates a smooth curve representing the data distribution.
    
    Args:
        data: A list or array of numerical data.
        title: Chart title.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.
    """
    data = np.array(data)
    kde = gaussian_kde(data)
    
    x_vals = np.linspace(min(data), max(data), 500)
    y_vals = kde(x_vals)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.fill_between(x_vals, y_vals, alpha=0.3) # Add fill for better visibility
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def hexbin_plot(x, y, gridsize: int = 20, title: str = "Hexbin Density Plot", 
                xlabel: str = "X", ylabel: str = "Y"):
    """
    Creates a hexagonal binning plot, useful for visualizing the density of 
    scatter data when there are too many points for a standard scatter plot.
    """
    plt.figure(figsize=(6, 5))
    plt.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1)
    cb = plt.colorbar(label='Count in Bin')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    plt.show()

# --- RELATIONAL PLOTS ---

def scatter_plot(x, y, title: str = "Scatter Plot", xlabel: str = "X", ylabel: str = "Y"):
    """
    Creates a standard 2D scatter plot to show the relationship between two variables.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.7) # Added alpha for overlapping points
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def scatter_3d(x, y, z, title: str = "3D Scatter Plot", 
            xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z"):
    """
    Creates a 3D scatter plot for visualizing three variables simultaneously.
    Points are colored based on their Z-value. 
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(x, y, z, c=z, cmap='plasma', marker='o')
    plt.colorbar(sc, label=zlabel) # Add colorbar to indicate Z values
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    plt.tight_layout()
    plt.show()


def scatter_matrix(data, labels=None, edgecolor='black', title="Scatter Matrix"):
    """
    Creates a matrix of scatter plots to visualize pairwise relationships 
    between multiple dimensions.
    
    Args:
        data: A list of lists or 2D array (Dimensions x Samples).
        labels: A list of string labels for each dimension.
        title: Chart title.
    """
    data = np.array(data)
    # Ensure data is (Dimensions x Samples). Transpose if passed as (Samples x Dimensions)
    if data.shape[0] > data.shape[1]: 
        # Heuristic: usually more samples than dimensions. 
        # If rows > cols, we assume input was (Samples x Features), so we Transpose.
        data = data.T
        
    dims = data.shape[0]

    fig, axes = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    for i in range(dims):
        for j in range(dims):
            ax = axes[i, j]
            # Diagonal: Histogram
            if i == j:
                ax.hist(data[i], bins=10, edgecolor=edgecolor)
            # Off-diagonal: Scatter plot
            else:
                ax.scatter(data[j], data[i], s=10, alpha=0.6)
            
            # Labels on the edges of the matrix
            if labels:
                if j == 0:
                    ax.set_ylabel(labels[i])
                if i == dims - 1:
                    ax.set_xlabel(labels[j])

            ax.grid(True, alpha=0.2)

    plt.suptitle(title, y=1.02, fontsize=16) # Title above the grid
    plt.tight_layout()
    plt.show()

# --- MATH FUNCTION PLOTTING ---

def plot_function(f, x_start: float, x_end: float, n: int = 1000, title: str = "Function Plot"):
    """
    Plots a mathematical function y = f(x) over a specified range.
    
    Args:
        f: A callable function (e.g., lambda x: x**2).
        x_start: Start of the x-axis range.
        x_end: End of the x-axis range.
        n: Number of points to generate (resolution).
        title: Chart title.
    """
    x = np.linspace(x_start, x_end, n)
    try:
        y = f(x)
    except:
        # Fallback if f cannot handle numpy arrays directly
        y = [f(val) for val in x]
        
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"y = f(x)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='black', linewidth=0.5) # x-axis line
    plt.axvline(0, color='black', linewidth=0.5) # y-axis line
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

try:
    from scipy.stats import gaussian_kde 
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    # Define placeholder/dummy if not installed, though generally a library assumes its dependencies
    gaussian_kde = None 
    Axes3D = None

# --- BASIC FUNCTION & DATA PLOTS ---

def plot_points(points: list[tuple], draw_line: bool = True, x_label: str = "X", 
                y_label: str = "Y", title: str = "Data Point Plot"):
    """
    Plots a list of (x, y) points, optionally connecting them with a line.
    
    Args:
        points: A list of (x, y) tuples.
        draw_line: If True, connects the points with a line (useful for time series/ordered data).
        x_label: Label for the X-axis.
        y_label: Label for the Y-axis.
        title: Chart title.
    """
    if not points: return
    xs, ys = zip(*points)

    plt.figure(figsize=(7, 5))
    
    plt.scatter(xs, ys, label="Data Points", marker='o')
    if draw_line:
        plt.plot(xs, ys, label="Connection Line", linestyle='--')
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()


def plot_function(f, x_min: float, x_max: float, resolution: int = 500, title: str = "Function Plot",
                xlabel: str = "x", ylabel: str = "f(x)"):
    """
    Plots a mathematical function y = f(x) over a specified range.
    (Integrated from plotter.py)
    
    Args:
        f: A callable function (e.g., lambda x: x**2) that accepts a NumPy array.
        x_min: Start of the x-axis range.
        x_max: End of the x-axis range.
        resolution: Number of points to generate.
        title: Chart title.
    """
    x = np.linspace(x_min, x_max, resolution)
    try:
        y = f(x)
    except Exception:
        # Fallback if f cannot handle numpy arrays directly (slower)
        y = np.array([f(val) for val in x])
        
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"y = f(x)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='black', linewidth=0.5) # x-axis line
    plt.axvline(0, color='black', linewidth=0.5) # y-axis line
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_multiple_functions(functions: list, x_min: float, x_max: float, 
                            resolution: int = 500, title: str = "Multiple Function Plot",
                            xlabel: str = "x", ylabel: str = "y"):
    """
    Plots a list of mathematical functions y = f(x) on the same axes.
    (Integrated from plotter.py)
    
    Args:
        functions: A list of callable functions.
        x_min: Start of the x-axis range.
        x_max: End of the x-axis range.
        resolution: Number of points to generate.
        title: Chart title.
    """
    x = np.linspace(x_min, x_max, resolution)
    plt.figure(figsize=(8, 5))
    
    for i, f in enumerate(functions):
        try:
            y = f(x)
        except Exception:
            y = np.array([f(val) for val in x])
            
        plt.plot(x, y, label=f"Function {i+1}")
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
