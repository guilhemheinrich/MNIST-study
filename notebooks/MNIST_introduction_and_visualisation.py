# %% [markdown]
# # MNIST Dataset Introduction and Visualization
#
# This notebook provides an introduction to the MNIST dataset and demonstrates various visualization techniques.

# %% [markdown]
# ## Dependencies
#
# Import required libraries and modules.

# %%
import sys
import os
from pathlib import Path


# Add project root to Python path
# Get current notebook path
notebook_path = Path.cwd()
# Go up one level to reach project root
project_root = notebook_path.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from src.utils.mnist_visualization import visualize_mnist_image
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ## Parameters
# We will define paramters for this notebook, such as the proportion of the dataset used for training and testing, to speed up generation and testing.

# %%
# Define parameters
RANDOM_SEED = 42
SAMPLES_PER_LABEL = 1000
CV_K_FOLD = 4
TRAIN_VALIDATION_SPLIT = 0.2

# %% [markdown]
# ## Loading MNIST Dataset
#
# Before loading the dataset, you need to download it first by running the `data/download_mnist.py` script. This only needs to be done once. The data files are not included in version control as it's a good practice to separate data from code versioning.
#
# The MNIST dataset is a large collection of handwritten digits that is commonly used for training various image processing systems. The database contains:
# - 60,000 training images
# - 10,000 testing images
# - Each image is a 28x28 grayscale image of a single digit (0-9)
# - Each pixel has a value between 0 (white) and 255 (black)
#
# After downloading, the dataset will be available in CSV format in the `data` directory, ready to be loaded.
#
# The script will prepare the data into two csv files:
# - `train.csv`: contains the training images and their labels
# - `test.csv`: contains the testing images and their labels


# %%
# Load training data
train_data = pd.read_csv(os.path.join(project_root, "data", "train.csv"))
X_train = train_data.iloc[:, 1:].values  # All columns except the first (labels)
y_train = train_data.iloc[:, 0].values  # First column (labels)

test_data = pd.read_csv(os.path.join(project_root, "data", "test.csv"))
X_test = test_data.iloc[:, 1:].values  # All columns except the first (labels)
y_test = test_data.iloc[:, 0].values  # First column (labels)
print(f"Dataset shape: {X_train.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# %% [markdown]
# ## Data Visualization
#
# Data visualization is the first step in any data science project. It helps us understand the data, identify patterns, and detect potential issues. Let's start by exploring our MNIST dataset through different visualization techniques.
#
# ### Sample Images Visualization
#
# Let's visualize one example for each digit class to get a sense of what our data looks like.

# %%
# Create a figure with subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Get one example for each digit (0-9)
for digit in range(10):
    # Find the first occurrence of this digit
    idx = np.where(y_train == digit)[0][0]
    visualize_mnist_image(X_train[idx], digit, ax=axes[digit])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Class Distribution
#
# Let's examine the distribution of classes in the dataset. This analysis is crucial for several reasons:
# - To verify if the dataset is balanced across all digits (0-9)
# - To identify any potential class imbalance that could bias our model training
# - To ensure we have enough examples of each digit for effective learning
# - To help determine if we need to apply any class balancing techniques
#
# **Note: Deep learning models are indeed sensitive to class imbalance. However, traditional machine learning models may be more robust — it depends on the specific model and how it's implemented.**

# %%
# Count occurrences of each digit
unique, counts = np.unique(y_train, return_counts=True)

# Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=unique, y=counts)
plt.title("Distribution of Digits in MNIST Dataset")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# The classes are relatively well balanced, and the number of samples per class is sufficient for effective learning (>> 50).
# ### Pixel Value Distribution with PCA
#
# Principal Component Analysis (PCA) is a dimensionality reduction technique that helps visualize high-dimensional data in a lower-dimensional space
# while preserving as much variance as possible. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis).
#
# Here, we use PCA to project our 784-dimensional data (28x28 pixels) onto a 2D plane. This visualization can give us insights into the complexity
# of our classification task:
# - If digits form well-separated clusters, the classification task should be relatively straightforward
# - If digits overlap significantly, the task may be more challenging and require more sophisticated models

# %%
# Perform PCA and plot
plt.figure(figsize=(12, 8))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(train_data.drop("label", axis=1))
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["label"] = train_data["label"]

# Plot PCA results with a categorical color palette
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="label",
    palette="tab10",  # Palette with good contrast for categorical data
    alpha=0.6,
    s=10,
)
plt.title("PCA of MNIST Training Set (First Two Components)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(title="Digit")

# Show all plots
plt.show()

# %% [markdown]
# We can see that :
# - The first two axis of the PCA capture 17% of the variance of the data. It is not a lot, but it is still better than nothing.
# - except for the digit 1, the other digits are quite blended in the same area, which can lead to a not trivial classification task.

# %% [markdown]
# ## Classification Task
#
# In this section, we will explore different machine learning approaches to solve the MNIST digit classification problem. We will implement and compare three different models:
# - Random Forest Classifier
# - PLS (Partial Least Squares) Classifier
# - Simple Perceptron
#
# Each model has its own strengths and characteristics that make it suitable for different aspects of the classification task. We will analyze their performance using standardized metrics to ensure a fair comparison.

# %% [markdown]
# ### Metrics & Learning
#
# When evaluating classification models, it's crucial to use appropriate metrics that provide a comprehensive view of the model's performance. Different metrics capture different aspects of the model's behavior:
#
# - **Accuracy**: Overall proportion of correctly classified samples
# - **Balanced Accuracy**: Accuracy adjusted for class imbalance
# - **Confusion Matrix**: Detailed breakdown of correct and incorrect predictions per class
# - **Classification Report**: Precision, recall, and F1-score for each class
#
# We will use the metrics module from our project to ensure consistent evaluation across all models.

# %%
from src.metrics import (
    print_classification_metrics,
    plot_classification_metrics,
    ClassificationMetrics,
    plot_classification_metrics_sequence,
)

# These functions will help us:
# - print_classification_metrics: Display standard classification metrics
# - plot_classification_metrics: Visualize predictions and actual values
# - ClassificationMetrics: Store training metrics for each epoch
# - plot_classification_metrics_sequence: Visualize training progress over epochs (for deep learning models)

# %% [markdown]
# ### Model Validation Strategy
#
# The metrics are used to effectively evaluate our models and prevent overfitting. We will use two validation approaches adapted to different model types.
#
# For Machine Learning algorithms, we will use k-fold [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), which enables:
# - More robust performance estimation
# - Full utilization of available data for both training and validation
# - Reduced variance in results
#
# For the Deep Learning Models (like theSimple Perceptron), we will use a traditional train/validation split because:
# - It allows real-time learning monitoring
# - It facilitates early stopping to prevent overfitting
# - It is better suited for models requiring iterative learning
#
# While these approaches differ, they share the same goal: ensuring our models generalize well to new data, by limiting the risk of overfitting and by optimizing the hyperparameters.
# %% [markdown]
# ### Preprocessing
# We will preprocess the data for two main reasons:
# - To balance the classes
# - To ease the learning process by reducing the size of the dataset
# %%
# Shuffle training data while keeping label correspondence
indices = np.arange(len(y_train))
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)
X_train_shuffled = X_train[indices]
y_train_shuffled = y_train[indices]

# Initialize lists to store selected samples
X_train_preprocessed = []
y_train_preprocessed = []

# Keep track of samples per label
samples_count = {i: 0 for i in range(10)}

# Select first SAMPLES_PER_LABEL samples for each label
for X_sample, y_label in zip(X_train_shuffled, y_train_shuffled):
    if samples_count[y_label] < SAMPLES_PER_LABEL:
        X_train_preprocessed.append(X_sample)
        y_train_preprocessed.append(y_label)
        samples_count[y_label] += 1

    # Check if we have enough samples for all labels
    if all(count >= SAMPLES_PER_LABEL for count in samples_count.values()):
        break

# Convert lists to numpy arrays
X_train_preprocessed = np.array(X_train_preprocessed)
y_train_preprocessed = np.array(y_train_preprocessed)

print(f"Preprocessed dataset shape: {X_train_preprocessed.shape}")


# %% [markdown]
# ## Random Forest Classifier
#
# The Random Forest Classifier is a powerful ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.


# %%
# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 3, 5],
}

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform GridSearchCV with 4-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=CV_K_FOLD,
    n_jobs=-1,
    scoring="accuracy",
    verbose=1,
)

# Fit the model
grid_search.fit(X_train_preprocessed, y_train_preprocessed)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Get best model
best_rf = grid_search.best_estimator_

# Make predictions on test set
y_pred = best_rf.predict(X_test)

# Print and plot classification metrics
print("\nTest Set Performance:")
print_classification_metrics(y_test, y_pred)
plot_classification_metrics(y_test, y_pred, "Random Forest Classification Results")

# %% [markdown]
# ### Results Analysis

#
# The best model has an accuracy of > 95% on the test set, as well as a balanced accuracy of > 95%.
# The confusion matrix is also quite good, with a low number of false positives and false negatives.
# The precision, recall and F1-score are also quite good for each class.

# %% [markdown]
# ## Partial Least Squares (PLS) Analysis

# PLS is a dimensionality reduction and regression technique that finds the directions of maximum covariance between X and Y.
# While it's primarily used for regression, we can adapt it for classification by using one-hot encoded target variables.

# %% [markdown]
# ### Pretreatment
# To use the PLS model, we need to convert the labels into a regression target, which means we will switch from a classification task to a regression task. To do so, we use a one-hot encoding of the labels, with as much dimensions as there are classes in the dataset and a single 1 value for the correct class.

# %%
# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train_preprocessed.reshape(-1, 1))

# Create class mapping
feature_names = encoder.get_feature_names_out(["digit"])
class_mapping = {i: int(name.split("_")[1]) for i, name in enumerate(feature_names)}

# %% [markdown]
# ### Setting up the PLS model

# %%
# Define parameter grid for cross-validation
param_grid = {"n_components": list(range(5, 21))}

# Initialize PLS model
pls = PLSRegression()

# Perform GridSearchCV with R2 scoring
grid_search = GridSearchCV(
    estimator=pls,
    param_grid=param_grid,
    cv=CV_K_FOLD,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

print("Starting grid search for PLS regression...")
grid_search.fit(X_train_preprocessed, y_train_encoded)

# Print results
print("\nGrid Search Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation R2 score: {grid_search.best_score_:.4f}")

# Plot cross-validation results
cv_results = pd.DataFrame(grid_search.cv_results_)
results_summary = cv_results[
    ["param_n_components", "mean_test_score", "std_test_score"]
]
results_summary = results_summary.sort_values("param_n_components")

plt.figure(figsize=(10, 6))
plt.errorbar(
    results_summary["param_n_components"],
    results_summary["mean_test_score"],
    yerr=results_summary["std_test_score"],
    fmt="o-",
)
plt.xlabel("Number of Components")
plt.ylabel("Cross-validation R2 Score")
plt.title("PLS Regression Performance vs Number of Components")
plt.grid(True)
plt.show()

# Use best model for predictions
best_pls = grid_search.best_estimator_

# %% [markdown]
# ### Predictions & Conclusion

# %%
# Make predictions on test set
y_test_pred_raw = best_pls.predict(X_test)
y_test_pred_classes = np.array(
    [class_mapping[i] for i in np.argmax(y_test_pred_raw, axis=1)]
)

# Print and plot classification metrics
print("\nClassification Metrics on Test Set:")
y_test_array = np.array(y_test)
y_test_pred_classes_array = np.array(y_test_pred_classes)
print_classification_metrics(y_test_array, y_test_pred_classes_array)
plot_classification_metrics(
    y_test_array, y_test_pred_classes_array, "PLS Classification Results"
)

# %% [markdown]
# ### Results Analysis
#
# The PLS model performs a bit lower than the Random Forest model, with an accuracy and a balanced accuracy a bit above 85%.
# It is not bad in itself, but produce three times more error than the Random Forest model.
