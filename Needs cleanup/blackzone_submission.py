import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
from sklearn import linear_model
from sklearn.externals import joblib

imgShape = nib.load('data/set_train/train_1.nii').shape

imgQuarterX = imgShape[0]/2
imgTopQuarterZ = imgShape[2]*3/4

# Fetch all directory listings of set_train
allImageSrc = glob.glob("data/set_train/*")
allTestSrc = glob.glob("data/set_test/*")

# Get the targets
with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets, dtype='int_')

# For now, try fitting the data with linear regression using
# only one dimension of the image data (i.e. by fixing 2 two variables
# and keeping 1 degree of freedom) 
#
# For this we'll be choosing 1/4th part of x-axis and 3/4th of y-axis, simply
# to catch the most interesting area of the brain
all_samples = []
for i in range(0,len(allImageSrc)):
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();
	# brain seen from top, center
	brain_slice = imgData[52:120, 65:150, 55:110, 0]
	black_pixels = 0
	'''
	for j in range(0,len(brain_slice)):
		for k in range(0, len(brain_slice[j])):
			for l in range(0, len(brain_slice[j][k])):
				if(brain_slice[j][k][l] < 450): black_pixels = black_pixels+1

	print "Percentage done: "+str((i*100)/len(allImageSrc))+" %" '''
	brain_slice = imgData[52:120, 65:150, 88, 0]
	black_pixels = 0
	for j in range(0,len(brain_slice)):
		for k in range(0, len(brain_slice[j])):
			if(brain_slice[j][k] < 750): black_pixels = black_pixels+1
	
	print "Percentage done: "+str((i*100)/len(allImageSrc))+" %"
	# Our 'feature' is how many black pixels we have:
	all_samples.append([black_pixels])

reg = linear_model.LinearRegression()
#reg = linear_model.Lasso(alpha = 0.1)

#print all_samples
#print targets
print "Fitting model"
reg.fit (all_samples, targets);
joblib.dump(reg, 'blackzone_submission.model')
#testPrediction = reg.predict(testData)[0][0];
#print "Prediction for datapoint "+str(i)+":"
#print testPrediction
#print "Actual value"
#print testTarget[0]
#print ""
#total_error += abs(testPrediction-testTarget[0])

print "Preparing test data"

all_test_samples = []
for i in range(0,len(allTestSrc)):
	img = nib.load(allTestSrc[i])
	imgData = img.get_data();
	# brain seen from top, center
	brain_slice = imgData[52:120, 65:150, 55:110, 0]
	black_pixels = 0
	'''
	for j in range(0,len(brain_slice)):
		for k in range(0, len(brain_slice[j])):
			for l in range(0, len(brain_slice[j][k])):
				if(brain_slice[j][k][l] < 450): black_pixels = black_pixels+1

	print "Percentage done: "+str((i*100)/len(allImageSrc))+" %" '''
	brain_slice = imgData[52:120, 65:150, 88, 0]
	black_pixels = 0
	for j in range(0,len(brain_slice)):
		for k in range(0, len(brain_slice[j])):
			if(brain_slice[j][k] < 750): black_pixels = black_pixels+1
	
	print "Percentage done: "+str((i*100)/len(allTestSrc))+" %"
	# Our 'feature' is how many black pixels we have:
	all_test_samples.append([black_pixels])

print "Making predictions"
predictions = map(int,map(round,reg.predict(all_test_samples).flatten().tolist()))
print predictions
with open('results.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(predictions[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()