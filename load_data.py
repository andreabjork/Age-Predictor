# A starting point for testing.

# NiBabel allows us to access .nii data files.
# Matplotlib can be used to view them.
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# Sci-Kit Learn offers a lot of Machine Learning stuff
import sklearn

# Put your training and test data from Kaggle into this path (relative to your
# project path). So './data/'' should contain 'targets.csv' and 
# the folders 'set_test' and 'set_train'.
# Don't forget to exclude them from the git repo.
data_path = './data/'