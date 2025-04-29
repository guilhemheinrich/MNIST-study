# %% [markdown]
# # Test Notebook
# 
# This notebook is used for testing and experimentation with the MNIST dataset.

# %% [markdown]
# ## Dependencies
# 
# Import required libraries and modules

# %%
import sys
import os

# Add project root to Python path
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %% [markdown]
# ## Reference to Main Notebook
# 
# This notebook is referenced in `MNIST_introduction_and_visualisation.py` for testing purposes. 

# %% tags=["remove_cell"]
a = 1
b = 2
c = a + b

# %% tags=["hide_input"]
print("Variables values:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c} (sum of a and b)")


# %% [markdown]
# ## Variables Reference
# 
# The values used in the previous cell are:
# - a = 1
# - b = 2
# - c = 3 (sum of a and b)


