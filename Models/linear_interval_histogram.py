import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pprint, pickle


# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets, dtype='int_')
lastTarget = targets[-1]

# Get the preprocessed features:
pkl_file = open('../Features/histogram.pkl', 'rb')
histograms = pickle.load(pkl_file)
pkl_file.close()

total_error = 0
for i in range(0, len(histograms)):
	trainData = np.array(histograms[:i]+histograms[i+1:])
	testData = np.array([histograms[i]])

	trainTargets = np.concatenate((targets[:i],targets[i+1:]))
	testTarget = targets[i]


	#reg = linear_model.LinearRegression()
	reg = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
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

