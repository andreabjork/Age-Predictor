import csv
import numpy as np
from sklearn import pipeline
from sklearn.feature_selection import *
from sklearn.neural_network import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from VotingRegressor import VotingRegressor
from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = [float(x[0]) for x in targets]

histograms = extractHistograms('../data/set_train',2500)
print "Shape of histograms:"
print np.array(histograms).shape

print "Estimating error:"

models = {
	"ExtraTreesRegressor (submission)" : pipeline.make_pipeline(
		ExtraTreesRegressor(n_estimators=100,warm_start=True,bootstrap=True,oob_score=True)
	),
	"VotingRegressor" : VotingRegressor(regs=[
			VotingRegressor(regs=[
					LinearRegression(),
					RidgeCV(),
					BayesianRidge(),
					ExtraTreesRegressor(n_estimators=100)
				], weights=[1,1,1,6])
			]*10
	)
}


for key, model in models.items():
	scores = cross_val_score(model, histograms, targets, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
	print "Average error: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)
