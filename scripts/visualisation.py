"""
Script to visualize MNIST images and label distribution.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mnist_visualization import visualize_mnist_image

# Load training data
df = pd.read_csv(project_root / "data" / "train.csv")

# Create a 4x3 grid of subplots
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle("First Example of Each Digit (0-9)", fontsize=16)

# Plot first example of each digit
for digit in range(10):
    row = digit // 3
    col = digit % 3
    ax = axes[row, col]
    
    # Get first example of current digit
    sample = df[df['label'] == digit].iloc[0]
    label = sample['label'].item()
    pixels = sample.drop('label').values.astype(int)
    
    # Visualize the image in the subplot
    visualize_mnist_image(pixels, label, ax=ax)

# Remove empty subplots
axes[3, 1].axis('off')
axes[3, 2].axis('off')

# Adjust layout
plt.tight_layout()

# Create distribution plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label')
plt.title("Distribution of Digits in Training Set")
plt.xlabel("Digit")
plt.ylabel("Count")

# Show all plots
plt.show() 