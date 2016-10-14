import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from sklearn import linear_model

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
# For now, try fitting the data with linear regression using
# only one dimension of the image data (i.e. by fixing 2 two variables
# and keeping 1 degree of freedom) 
#
# For this we'll be choosing 1/4th part of x-axis and 3/4th of y-axis, simply
# to catch the most interesting area of the brain
hist_max_value = 4500
total_histogram = [0]*hist_max_value
histograms = []
for i in range(0,len(allImageSrc)):
	print str((i*100)/len(allImageSrc))+"% working..."
	hist = [0]*hist_max_value
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();
	for val in imgData.flatten().tolist():
		if val < hist_max_value:
			total_histogram[val] += 1
			hist[val] += 1
	histograms.append(hist)
hist_cat = range(hist_max_value)
width = 1/1.5
#plt.plot(hist_cat,total_histogram)
plt.bar(hist_cat,total_histogram,width,color="blue")
plt.show()
total_error = 0
for i in range(0, len(histograms)):
	trainData = np.array(histograms[:i]+histograms[i+1:])
	testData = np.array([histograms[i]])

	trainTargets = np.concatenate((targets[:i],targets[i+1:]))
	testTarget = targets[i]


	reg = linear_model.LinearRegression()
	reg.fit (trainData, trainTargets);

	testPrediction = reg.predict(testData)[0][0];
	print "Prediction for datapoint "+str(i)+":"
	print testPrediction
	print "Actual value"
	print testTarget[0]
	print ""
	total_error += abs(testPrediction-testTarget[0])

avg_error_1 = total_error/len(histograms)
print "Average error (method 1) "+str(avg_error_1)

