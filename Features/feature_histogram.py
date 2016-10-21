import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Fetch all directory listings of set_train
allImageSrc = glob.glob("../data/set_train/*")

# For now, try fitting the data with linear regression using
# only one dimension of the image data (i.e. by fixing 2 two variables
# and keeping 1 degree of freedom) 
#
# For this we'll be choosing 1/4th part of x-axis and 3/4th of y-axis, simply
# to catch the most interesting area of the brain
hist_max_value = 1000
histograms = []
for i in range(0,len(allImageSrc)):
	print str((i*100)/len(allImageSrc))+"% working..."
	hist = [0]*hist_max_value
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();
	for val in imgData.flatten().tolist():
		if val < hist_max_value:
			hist[val] += 1
	histograms.append(hist)

output = open('histogram_1000.pkl', 'wb')
pickle.dump(histograms, output)
output.close()

