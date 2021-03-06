import csv
import numpy as np
from sklearn import pipeline
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.model_selection import cross_val_score

from Features.extract_features import *
from Models.VotingRegressor import VotingRegressor

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array([float(x[0]) for x in targets])

histograms = np.array(extractHistograms('../data/set_train',2500))

print "Training model"
model = pipeline.make_pipeline(
			VarianceThreshold(threshold=10),
			VotingRegressor(regs=[
				RandomForestRegressor(n_estimators=100),
				AdaBoostRegressor(n_estimators=100),
				BaggingRegressor(n_estimators=100)
			])
		)
model.fit(histograms,targets)

print "Testing model"
testData = np.array(extractHistograms('../data/set_test',2500))
predictions = model.predict(testData).flatten().tolist()
print predictions
with open('VotingRegressor.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(predictions[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()