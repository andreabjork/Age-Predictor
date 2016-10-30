import csv
import numpy as np
from random import shuffle
from sklearn import pipeline
from sklearn.feature_selection import *
from sklearn.neural_network import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from VotingRegressor import VotingRegressor
from LooRegressor import LooRegressor
from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = [float(x[0]) for x in targets]

data = extractAverages('../data/set_train')
print "Shape of data:"
print np.array(data).shape

print "Estimating error:"

models = {
	"ExtraTreesRegressor (submission)" : pipeline.make_pipeline(
		ExtraTreesRegressor(n_estimators=100,bootstrap=True,oob_score=True)
	),
	"Linear (Poly = 2)" : pipeline.make_pipeline(
			PolynomialFeatures(2),
			LinearRegression()
	),
	"Ridge (Poly = 2)" : pipeline.make_pipeline(
			PolynomialFeatures(2),
			Ridge()
	),
	"RidgeCV (Poly = 2)" : pipeline.make_pipeline(
			PolynomialFeatures(2),
			RidgeCV()
	),
	"ExtraTreesRegressor (Poly = 2)" : pipeline.make_pipeline(
			PolynomialFeatures(2),
			ExtraTreesRegressor(n_estimators=100,bootstrap=True,oob_score=True)
	)
	#"LooRegressor (RidgeCV)" : LooRegressor(RidgeCV())
}

hist_targ = zip(data,targets)
shuffle(hist_targ)
X_shuffled, y_shuffled = zip(*hist_targ)

n_test_data = 10
n_tests = 10
errors = {}
for i in range(n_tests):
	X_train = X_shuffled[:-n_test_data]
	y_train = y_shuffled[:-n_test_data]
	X_test = X_shuffled[-n_test_data:]
	y_test = y_shuffled[-n_test_data:]
	for key, model in models.items():
		model.fit(X_train,y_train)
		prediction = np.array(model.predict(X_test))
		truth = np.array(y_test)
		difference = np.absolute(prediction-truth)
		average_error = (sum(difference)*1.)/len(difference)
		if key in errors:
			errors[key] += (average_error*1.)/n_tests
		else:
			errors[key] = (average_error*1.)/n_tests

for key, error in sorted(errors.items()):
	print "Average error: %f [%s]"%(error,key)

