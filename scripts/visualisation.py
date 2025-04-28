"""
Script to visualize MNIST images.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.utils.mnist_visualization import visualize_mnist_image

# Load a sample from the training data
df = pd.read_csv(project_root / "data" / "train.csv")
sample = df.iloc[0]

# Extract label and pixels
label_series = sample['label']
label = label_series.item()  # Convert Series to int using .item()
pixels = sample.drop('label').values.astype(int)  # Convert to numpy array of ints

# Visualize the image
visualize_mnist_image(pixels, label) 