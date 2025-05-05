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
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

# %% [markdown]
# ## Parameters
# We will define paramters for this notebook, such as the proportion of the dataset used for training and testing, to speed up generation and testing.

# %%
# Define parameters
RANDOM_SEED = 42
SAMPLES_PER_LABEL = 1000
# Number of folds for cross-validation
CV_K_FOLD = 4
# Neural network training parameters
TRAIN_VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.01
NUM_EPOCHS = 50  # Maximum number of epochs for training
PATIENCE = 5  # Number of epochs to wait for improvement before early stopping
# Kernel size for convolutional layers
KERNEL_SIZE = 3
# Stride for convolutional layers
STRIDE = 1
# Padding for convolutional layers
PADDING = 2
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
# ## Machine Learning
# ### Random Forest Classifier
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
# #### Results Analysis

#
# The best model has an accuracy of > 95% on the test set, as well as a balanced accuracy of > 95%.
# The confusion matrix is also quite good, with a low number of false positives and false negatives.
# The precision, recall and F1-score are also quite good for each class.

# %% [markdown]
# ### Partial Least Squares (PLS) Analysis

# PLS is a dimensionality reduction and regression technique that finds the directions of maximum covariance between X and Y.
# While it's primarily used for regression, we can adapt it for classification by using one-hot encoded target variables.

# %% [markdown]
# #### Pretreatment
# To use the PLS model, we need to convert the labels into a regression target, which means we will switch from a classification task to a regression task. To do so, we use a one-hot encoding of the labels, with as much dimensions as there are classes in the dataset and a single 1 value for the correct class.

# %%
# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train_preprocessed.reshape(-1, 1))

# Create class mapping
feature_names = encoder.get_feature_names_out(["digit"])
class_mapping = {i: int(name.split("_")[1]) for i, name in enumerate(feature_names)}

# %% [markdown]
# #### Setting up the PLS model

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
# #### Predictions & Conclusion

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
# #### Results Analysis
#
# The PLS model performs a bit lower than the Random Forest model, with an accuracy and a balanced accuracy a bit above 85%.
# It is not bad in itself, but produce three times more error than the Random Forest model.
# %% [markdown]
# ## Deep Learning
# ### Data Preparation
#
# We need to:
# 1. Convert our NumPy data to PyTorch tensors and normalize them
# 2. Split training data into train and validation sets
# 3. Create data loaders for each set

# %%
# Convert preprocessed data to PyTorch tensors and normalize
X_preprocessed_tensor = torch.FloatTensor(X_train_preprocessed) / 255.0
y_preprocessed_tensor = torch.LongTensor(y_train_preprocessed)

# Split preprocessed data into train and validation sets
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
    X_preprocessed_tensor,
    y_preprocessed_tensor,
    test_size=TRAIN_VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    stratify=y_preprocessed_tensor,
)

# Convert test data to tensors (will only be used for final evaluation)
X_test_tensor = torch.FloatTensor(X_test) / 255.0
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# %% [markdown]
# ### Perceptron
#
# The Perceptron is the simplest deep learning model. It consists of a single layer of neurons that takes as input
# the image pixels (784 values) and outputs 10 values (one for each class).
#
# #### Model Implementation


# %%
class SimplePerceptron(nn.Module):
    """Simple Perceptron model for MNIST classification.

    Args:
        input_size: Number of input features (784 for MNIST)
        num_classes: Number of output classes (10 for MNIST)
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.linear(x)


# %% [markdown]
# #### Model, criterion and optimizer
# We use a CrossEntropyLoss as criterion to compute the loss between the predicted and the true labels. The same logic is applied form the PLS model, where we need to convert the classification to a regression one.
# %%
# Initialize model, loss function and optimizer
model = SimplePerceptron(input_size=784, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
summary(model)
# %%[markdown]
# #### Training loop
# We will use early stopping based on validation loss to prevent overfitting.

# %% tags=["hide_output"]
# Training parameters
best_val_loss = float("inf")
patience_counter = 0
metrics = []

# Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_train_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.numpy())
            val_labels.extend(labels.numpy())

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = np.mean(np.array(val_preds) == np.array(val_labels))

    # Store metrics
    metrics.append(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        }
    )

    # Print progress
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("----------------------------------------")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model state
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
# %% [markdown]
# We plot the training progress to see when the model is overfitting.
# %%
# Plot training progress
epochs = [m["epoch"] for m in metrics]
train_losses = [m["train_loss"] for m in metrics]
val_losses = [m["val_loss"] for m in metrics]
val_accuracies = [m["val_accuracy"] for m in metrics]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Final evaluation
# We load the best model and evaluate it on the test set.
# %%
# Load best model for final evaluation
model.load_state_dict(best_model_state)

print("\nFinal Test Set Performance:")
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds.extend(predicted.numpy())
        test_labels.extend(labels.numpy())
# %% [markdown]
# #### Results Analysis
# We print the results and plot the confusion matrix.
# %%
print_classification_metrics(np.array(test_labels), np.array(test_preds))
plot_classification_metrics(
    np.array(test_labels), np.array(test_preds), "Perceptron Classification Results"
)


# %% [markdown]
# The results are comparable to the Random Forest model, with an accuracy and a balanced accuracy a bit above 91%.
# %% [markdown]
# ### Convolutional network
#
# The convolutional network is a more complex model that is able to learn spatial hierarchies of features from the image.
# Here it is used to take into account the spatial structure of the image.
#
# #### Model Implementation
# In this case, we will use two convolutional layers, with a ReLU activation function and a max pooling layer after each convolutional layer.
# We use 2d convolutional layers, as the data being an image possesses two spatial dimensions.
# %%
class CNNModel(nn.Module):
    """CNN model with two convolutional layers for MNIST image classification

    Args:
        input_channels: Number of input channels (1 for grayscale images)
        num_classes: Number of output classes (10 for MNIST digits)
        kernel_size: Size of the convolutional kernel
        stride: Stride of the convolution
        padding: Padding added to the input
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        kernel_size: int = KERNEL_SIZE,
        stride: int = STRIDE,
        padding: int = PADDING,
    ):
        super().__init__()

        # First convolutional layer: in_channels=1 (grayscale), out_channels=16
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Second convolutional layer: in_channels=16, out_channels=32
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculate the size of the flattened features after conv layers
        # Input size: 28x28
        # We compute the size of the output only on one dimension, as the image is a square.
        # After conv1: ((28 + 2*padding - kernel_size) // stride + 1) // 2
        # After conv2: ((previous + 2*padding - kernel_size) // stride + 1) // 2
        conv1_out = ((28 + 2 * padding - kernel_size) // stride + 1) // 2
        conv2_out = ((conv1_out + 2 * padding - kernel_size) // stride + 1) // 2
        # We multiply the number of channels by the size of the output of the last convolutional layer.
        flattened_size = 32 * conv2_out * conv2_out

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 1, 28, 28)
        x = self.conv1(x)
        # Shape: (batch_size, 16, 14, 14)
        x = self.conv2(x)
        # Shape: (batch_size, 32, 7, 7)

        # Flatten: (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)

        # Output: (batch_size, num_classes)
        return self.fc(x)


# #### Model, criterion and optimizer
# We use a CrossEntropyLoss as criterion to compute the loss between the predicted and the true labels, just as the perceptron.
# %%
# Initialize model, loss function and optimizer
model = CNNModel(
    input_channels=1,
    num_classes=10,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
summary(model)

# %% [markdown]
# #### Training loop
# We will use early stopping based on validation loss to prevent overfitting.
# Notice that the input size is 28x28, which is the size of the image, and that this shape is indicated to the network by the view function.
# %%
# Training parameters
best_val_loss = float("inf")
patience_counter = 0
metrics = []

# Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        # Reshape inputs to (batch_size, 1, 28, 28)
        inputs = inputs.view(-1, 1, 28, 28)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Reshape inputs to (batch_size, 1, 28, 28)
            inputs = inputs.view(-1, 1, 28, 28)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.numpy())
            val_labels.extend(labels.numpy())

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    # Store metrics
    metrics.append(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        }
    )

    # Print progress
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(
        f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}"
    )
    print("----------------------------------------")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model state
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break


# %%
# Plot training progress
epochs = [m["epoch"] for m in metrics]
train_losses = [m["train_loss"] for m in metrics]
val_losses = [m["val_loss"] for m in metrics]
train_accuracies = [m["train_accuracy"] for m in metrics]
val_accuracies = [m["val_accuracy"] for m in metrics]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
# %% [markdown]
# Here again, we will stop the training when the validation loss is no longer decreasing, to avoid overfitting.
# %%
# Load best model for final evaluation
model.load_state_dict(best_model_state)

model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Reshape inputs to (batch_size, 1, 28, 28)
        inputs = inputs.view(-1, 1, 28, 28)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds.extend(predicted.numpy())
        test_labels.extend(labels.numpy())

# %% tags=["hide_input"]
# Print and plot classification metrics
print_classification_metrics(np.array(test_labels), np.array(test_preds))
plot_classification_metrics(
    np.array(test_labels), np.array(test_preds), "CNN Classification Results"
)

# %% [markdown]
# #### Results
# The results are better to the random forest model (our previous best model), with an accuracy and a balanced accuracy around 98%. It means that it predicts wrongly more than twice less often than the random forest classifier.
# The draw back of the convolutional network is that it is more difficult to understand the decision of the model, as it is a black box. It may need more time to train, more data to b properly trained, and we need extra reasearch to knwo if our model is a good representant of the CNN models.
# Said otherwise, we have more hyperparameters to tune, and those tuning requires more time.
