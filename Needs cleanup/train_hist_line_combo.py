import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib

imgShape = nib.load('data/set_train/train_1.nii').shape

imgQuarterX = imgShape[0]/2
imgTopQuarterZ = imgShape[2]*3/4

# Fetch all directory listings of set_train
allImageSrc = glob.glob("data/set_train/*")

# Get the targets
with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets, dtype='int_')
lastTarget = targets[-1]

hist_max_value = 4500
histograms = []
all_samples = []
n_samples = len(allImageSrc);
print "Preparing the train data"
for i in range(0,n_samples):
	print str((i*100)/n_samples)+"% working..."
	hist = [0]*hist_max_value
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();

	## for histogram
	for val in imgData.flatten().tolist():
		if val < hist_max_value:
			hist[val] += 1
	histograms.append(hist)

	## for brainline
	brainLine = imgData[imgQuarterX, :, imgTopQuarterZ, 0]
	brainLine = np.array(brainLine, dtype='int_')
	all_samples.append(brainLine)

print "Fitting the model"
trainData_h = np.array(histograms)
trainData_l = np.array(all_samples)
trainTargets = targets

reg_h = linear_model.LinearRegression()
reg_h.fit (trainData_h, trainTargets);
joblib.dump(reg_h, 'histogram.model')

reg_l = linear_model.LinearRegression()
reg_l.fit (trainData_l, trainTargets);
joblib.dump(reg_l, 'brainline.model')