# MNIST Dataset

Original files (`.gz`) were downloaded from the CVDF Foundation official repository: [cvdfoundation/mnist](https://github.com/cvdfoundation/mnist).

## Directory Structure

### Raw Data (`./raw`)
- `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`: Original training data
- `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`: Original test data

### Processed Data
- `train.csv`, `test.csv`: Files generated from original data

## CSV Files Format

- First column: 'label' (0-9)
- Following columns: 784 pixels in grayscale (0-255)