import nibabel as nib
import matplotlib.pyplot as plt
import glob
import sys
import csv
import numpy as np

with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets, dtype='int_')


allImageSrc = glob.glob("data/set_train/*")
all_samples = []
'''
	Various definitons of targets (you can add to this)	
'''

# BLACKZONE

if(sys.argv[1] == "blackzone2D"):
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

		# Our 'feature' is how many black pixels we have:
		all_samples.append([black_pixels])

# BLACKZONE HISTOGRAM


if(sys.argv[1] == "blackhisto2D"):
	print "not configured"

# HISTOGRAM


if(sys.argv[1] == "histo"):
	hist_max_value = 4500
	total_histogram = [0]*hist_max_value
	histograms = []
	for i in range(0,len(allImageSrc)):
		print str((i*100)/len(allImageSrc))+"% working..."
		hist = [0]*hist_max_value
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		for val in imgData.flatten().tolist():
			if val < hist_max_value:
				total_histogram[val] += 1
				hist[val] += 1
		histograms.append(hist)

	all_samples = histograms

# 1D FROM BRAIN MRI


if(sys.argv[1] == "brainline"):
	imgShape = nib.load('data/set_train/train_1.nii').shape
	imgQuarterX = imgShape[0]/2
	imgTopQuarterZ = imgShape[2]*3/4
	for i in range(0,len(allImageSrc)):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
	
		# Get the age and the center img slice:
		brainLine = imgData[imgQuarterX, :, imgTopQuarterZ, 0]
		brainLine = np.array(brainLine, dtype='int_')
		all_samples.append(brainLine)









'''
	---------------------------------------
'''

plt.scatter(all_samples, targets)
plt.show()