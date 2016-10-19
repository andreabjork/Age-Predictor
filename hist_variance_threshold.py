import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
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

hist_t = np.array([[histograms[x][y] for x in range(histograms.shape[0])] for y in range(histograms.shape[1])])
variances = sorted([np.var(hist_t[x]) for x in range(hist_t.shape[0])])
minVar = variances[-600]
sel = VarianceThreshold(threshold=minVar)
print "Fitting the model"
trainData = sel.fit_transform(np.array(histograms))
trainTargets = targets

model = pipeline.make_pipeline(PolynomialFeatures(2),linear_model.LinearRegression())
model.fit (trainData, trainTargets);
joblib.dump(model, 'histogram_varthresh.model')
joblib.dump(sel, 'histogram_varthresh.threshold')