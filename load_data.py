# A starting point for testing.

# NiBabel allows us to access .nii data files.
# Matplotlib can be used to view them.
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import feature_extraction as ftex
# Sci-Kit Learn offers a lot of Machine Learning stuff
import sklearn

# The (trained) model can be saved to disk to avoid retraining on startup using 'joblib'.
# joblib.dump(clf, 'filename.pkl') saves to disk
# clf = joblib.load('filename.pkl')  restores it.
# See more: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence
from sklearn.externals import joblib

# Put your training and test data from Kaggle into this path (relative to your
# project path). So './data/'' should contain 'targets.csv' and 
# the folders 'set_test' and 'set_train'.
# Don't forget to exclude them from the git repo.
data_path = './data/'

# Read targets.csv into a list
age = []
with open(data_path+"targets.csv") as f:
	age = [int(x.strip('\n')) for x in f.readlines()]

# Function to display row of image slices
def show_slices(slices):
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")

# Simple test feature: Try averaging the intensity value of each person's image.
average = ftex.extract_average(data_path,len(age))

# Plot the age - average pairs
plt.plot(age,average)
plt.savefig("avg_test.png")
