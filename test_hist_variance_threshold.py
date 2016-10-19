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

allTestSrc = glob.glob("data/set_test/*")

model = joblib.load('hist_variance_threshold.model')
sel = joblib.load('hist_variance_threshold.threshold')

hist_max_value = 4500
histograms = []
n_samples = len(allTestSrc);
print "Preparing data"
for i in range(0,n_samples):
	print str((i*100)/n_samples)+"% working..."
	hist = [0]*hist_max_value
	img = nib.load(allTestSrc[i])
	imgData = img.get_data();

	## for histogram
	for val in imgData.flatten().tolist():
		if val < hist_max_value:
			hist[val] += 1
	histograms.append(hist)
testData = sel.transform(np.array(histograms))

print "Making predictions predictions"

predictions = model.predict(testData)

print predictions

print "Storing predictions"
with open('results_hist_variance_threshold.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(int(round(predictions[i])))
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close() 