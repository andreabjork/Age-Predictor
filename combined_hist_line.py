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

hist_max_value = 4500
histograms = []
all_samples = []
n_samples = len(allImageSrc);
for i in range(0,len(allImageSrc)):
	print str((i*100)/len(allImageSrc))+"% working..."
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


total_error_h = 0
total_error_l = 0
total_error_c = 0
for i in range(0, len(histograms)):
	trainData_h = np.array(histograms[:i]+histograms[i+1:])
	testData_h = np.array([histograms[i]])

	trainData_l = np.array(all_samples[:i] + all_samples[i+1:])
	testData_l = np.array([all_samples[i]])

	trainTargets = np.concatenate((targets[:i],targets[i+1:]))
	testTarget = targets[i]


	reg_h = linear_model.LinearRegression()
	reg_h.fit (trainData_h, trainTargets);

	reg_l = linear_model.LinearRegression()
	reg_l.fit (trainData_l, trainTargets);

	testPrediction_h = reg_h.predict(testData_h)[0][0];
	testPrediction_l = reg_l.predict(testData_l)[0][0];
	testPrediction_c = (testPrediction_h+testPrediction_l)/2.
	print "Prediction for datapoint "+str(i)+":"
	print "histogram :"+str(testPrediction_h)
	print "brainline :"+str(testPrediction_l)
	print "combined  :"+str(testPrediction_c)
	print "Actual value"
	print testTarget[0]
	print ""
	total_error_h += abs(testPrediction_h-testTarget[0])
	total_error_l += abs(testPrediction_l-testTarget[0])
	total_error_c += abs(testPrediction_c-testTarget[0])

avg_error_h = total_error_h/n_samples
avg_error_l = total_error_l/n_samples
avg_error_c = total_error_c/n_samples
print "Average error (histogram): "+str(avg_error_h)
print "Average error (brainline): "+str(avg_error_l)
print "Average error (combined):  "+str(avg_error_c)