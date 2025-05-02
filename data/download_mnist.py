import numpy as np
import pandas as pd
import os
import gzip
import urllib.request
from pathlib import Path


def download_mnist() -> None:
    """Download MNIST dataset files from CVDF Foundation repository.

    Downloads the following files to data/raw directory:
    - train-images-idx3-ubyte.gz
    - train-labels-idx1-ubyte.gz
    - t10k-images-idx3-ubyte.gz
    - t10k-labels-idx1-ubyte.gz

    Creates data/raw directory if it doesn't exist.
    """
    # URLs for MNIST dataset files
    urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    ]

    # Create data/raw directory if it doesn't exist
    raw_dir = Path(__file__).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download each file
    for url in urls:
        filename = url.split("/")[-1]
        filepath = raw_dir / filename

        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists, skipping download")


def load_mnist_images(filename):
    """Load and decompress MNIST images"""
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # type: ignore[arg-type]
    return data.reshape(-1, 784)


def load_mnist_labels(filename):
    """Load and decompress MNIST labels"""
    with gzip.open(filename, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)  # type: ignore[arg-type]


def prepare_mnist():
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "raw")

    files = {
        "train_images": os.path.join(raw_dir, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(raw_dir, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz"),
    }

    print("Loading and preparing data...")

    # Loading data from files in raw/
    X_train = load_mnist_images(files["train_images"])
    y_train = load_mnist_labels(files["train_labels"])
    X_test = load_mnist_images(files["test_images"])
    y_test = load_mnist_labels(files["test_labels"])

    print("Creating CSV files...")

    train_df = pd.DataFrame(X_train)
    train_df.insert(0, "label", y_train)
    train_df.to_csv(os.path.join(base_dir, "train.csv"), index=False)

    test_df = pd.DataFrame(X_test)
    test_df.insert(0, "label", y_test)
    test_df.to_csv(os.path.join(base_dir, "test.csv"), index=False)

    print(f"Data successfully saved!")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    download_mnist()
    prepare_mnist()
