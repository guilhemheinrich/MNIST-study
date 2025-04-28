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
from sklearn.decomposition import PCA
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

# Perform PCA and plot
plt.figure(figsize=(12, 8))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop('label', axis=1))
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['label'] = df['label']

# Plot PCA results with a categorical color palette
sns.scatterplot(
    data=df_pca,
    x='PC1',
    y='PC2',
    hue='label',
    palette='tab10',  # Palette with good contrast for categorical data
    alpha=0.6,
    s=10
)
plt.title("PCA of MNIST Training Set (First Two Components)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(title='Digit')

# Show all plots
plt.show() 