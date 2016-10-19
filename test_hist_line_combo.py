import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib

allTestSrc = glob.glob("data/set_test/*")

reg_h = joblib.load('histogram.model')
reg_l = joblib.load('brainline.model')

n_samples = len(allTestSrc);
predictions = []
print "Generating predictions"
for i in range(0,n_samples):
	print str((i*100)/n_samples)+"% working..."
	hist = [0]*hist_max_value
	img = nib.load(allTestSrc[i])
	imgData = img.get_data();

	## for histogram
	for val in imgData.flatten().tolist():
		if val < hist_max_value:
			hist[val] += 1
	testData_h = np.array([hist])
	testPrediction_h = reg_h.predict(testData_h)[0][0];

	## for brainline
	brainLine = imgData[imgQuarterX, :, imgTopQuarterZ, 0]
	brainLine = np.array(brainLine, dtype='int_')
	testData_l = np.array([brainLine])

	testPrediction_l = reg_l.predict(testData_l)[0][0];
	testPrediction_c = int(round((testPrediction_h+testPrediction_l)/2.))
	predictions.append(testPrediction_c)

print "Storing predictions"
print predictions
with open('results_hist_line_combo.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(predictions[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()