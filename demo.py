import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg

with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

imgShape = nib.load('data/set_train/train_1.nii').shape
print imgShape
imgCenterX = imgShape[1]/2
imgCenterY = imgShape[1]/2
imgCenterZ = imgShape[1]/2
allImageSrc = glob.glob("data/set_train/*")


for i in range(0,len(allImageSrc)):
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();
	
	# Get the age and the center img slice:
	age = targets[i][0]
	brainSlice = imgData[50:128, 50:128, imgCenterZ, 0]
	brainSliceInv = nlg.inv(brainSlice)

	