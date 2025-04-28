"""
MNIST visualization utilities.

This module provides functions to visualize MNIST dataset images.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence
from numpy.typing import ArrayLike

def visualize_mnist_image(
    pixels: Sequence[int] | ArrayLike, 
    label: int | None = None,
    ax: plt.Axes | None = None
) -> None:
    """Visualize a single MNIST image from its pixel values.

    Args:
        pixels: Sequence of 784 pixel values (0-255) representing the image
        label: Optional label (0-9) to display as the title
        ax: Optional matplotlib Axes to plot on. If None, creates a new figure.

    Raises:
        ValueError: If the length of pixels is not 784
    """
    if len(pixels) != 784:
        raise ValueError("Input must contain exactly 784 pixels")

    # Reshape the 1D array into a 28x28 image
    image = np.array(pixels).reshape(28, 28)
    
    # Create the plot
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    
    sns.heatmap(
        image,
        cmap='gray_r',  # Reversed grayscale for better visibility
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    
    # Add title if label is provided
    if label is not None:
        ax.set_title(f"Label: {label}")
    
    if ax is None:
        plt.tight_layout()
        plt.show() 