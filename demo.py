import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
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
all_samples = []
for i in range(0,len(allImageSrc)):
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();
	
	# Get the age and the center img slice:
	brainLine = imgData[imgQuarterX, :, imgTopQuarterZ, 0]
	brainLine = np.array(brainLine, dtype='int_')
	all_samples.append(brainLine)

total_error = 0
for i in range(0, len(all_samples)):
	trainData = all_samples[:i] + all_samples[i+1:]
	testData = all_samples[i]

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

avg_error_1 = total_error/len(all_samples)
print "Average error (method 1) "+str(avg_error_1)




'''
Things that I could plot:

* 2 hluti á sömu mynd:
predicted value og actual value, þá bara sem fall af myndinni
Inn á þessa mynd mætti teikna fallið sem við fáum með prediction módelinu

* Reiknað errorinn (óþarft að plotta)

----------
Þegar það er búið:

Setja upp einfalt forrit sem plottar f(x) = aldur þar
sem x er feature.

Plotta svo upp öll möguleg svoleiðis feature og sjá
hvar fylgnin er.

Hvernig sjáum við fylgni?

Til dæmis:




'''

