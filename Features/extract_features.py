import os
import inspect
import nibabel as nib
import glob
import pickle
import math
from printProgress import printProgress
featuresDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# imgDirFullPath must be a string and maxValue must be an integer
def extractHistograms(imgDirFullPath, maxValue = 4000, nPartitions = -1):
	if nPartitions == -1: nPartitions=maxValue
	hist_max_value = int(maxValue)
	clusterSize = math.ceil((maxValue*1.)/nPartitions)
	imgPath = os.path.join(imgDirFullPath,"*")
	print "\nhist_max_value = "+str(hist_max_value)+"  "+str(type(hist_max_value))
	print "imgPath = "+imgPath+""
	print "number of classes = "+str(nPartitions)+"\n\n"

	outputFileName = os.path.join(featuresDir,"histograms_"+str(nPartitions)+"-"+str(hist_max_value)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		hist = [0]*nPartitions
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		## for histogram
		for val in imgData.flatten().tolist():
			if val < hist_max_value:
				c = int(val/clusterSize)
				hist[c] += 1
		histograms.append(hist)
		printProgress(i, n_samples)

	printProgress(n_samples, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"
	return histograms

def extractBrainSliceHistograms(imgDirFullPath, numSlices):
	if nPartitions == -1: nPartitions=maxValue
	hist_max_value = int(maxValue)
	clusterSize = math.ceil((maxValue*1.)/nPartitions)
	imgPath = os.path.join(imgDirFullPath,"*")
	print "\nhist_max_value = "+str(hist_max_value)+"  "+str(type(hist_max_value))
	print "imgPath = "+imgPath+""
	print "number of classes = "+str(nPartitions)+"\n\n"

	outputFileName = os.path.join(featuresDir,"brainslicehistograms_"+str(nPartitions)+"-"+str(hist_max_value)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		hist = [0]*nPartitions
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		## for histogram
		for val in imgData.flatten().tolist():
			if val < hist_max_value:
				c = int(val/clusterSize)
				hist[c] += 1
		histograms.append(hist)
		printProgress(i, n_samples)

	printProgress(n_samples, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"
	return histograms

def extractAverages(imgDirFullPath):
	imgPath = os.path.join(imgDirFullPath,"*")
	print "imgPath = "+imgPath+""

	outputFileName = os.path.join(featuresDir,"averages_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		averages = pickle.load(save)
		save.close()
		return averages

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)

	averages = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data().flatten().tolist();
		avg = (sum(imgData)*1.)/len(imgData)
		averages.append([avg])
		printProgress(i, n_samples)

	printProgress(n_samples, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(averages,output)
	output.close()
	print "Done"
	return averages

def flattenedImages(imgDirFullPath):
	imgPath = os.path.join(imgDirFullPath,"*")
	print "imgPath = "+imgPath+""

	outputFileName = os.path.join(featuresDir,"flattened_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		images = pickle.load(save)
		save.close()
		return images

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)

	images = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data().flatten().tolist();
		images.append(imgData)
		printProgress(i, n_samples)

	printProgress(n_samples, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(images,output)
	output.close()
	print "Done"
	return images

# Extract no. of voxels w. intensity above 1000, 1200 and 1400 in each of 4 segments.
# The segments are front-upper, back-upper, front-lower and back-lower.
# The data is pickled into a list of patients which for each patient contains
# a length 4 list with the data for each brain segment.
def extractWhiteMatter(imgDirFullPath):
	imgPath = os.path.join(imgDirFullPath,"*")
	print "imgPath = "+imgPath+""

	outputFileName = os.path.join(featuresDir,"whitematter_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		whitematter = pickle.load(save)
		save.close()
		return averages

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)

	whitematter = [] #averages = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"

	thresh = 1000;
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data() #.flatten().tolist();
		[xlim, ylim, zlim, Ilim] = imgData.shape
		[x_border, z_border] = [int(xlim/2), int(zlim/2)] 
		brainregions = [];
		# back-lower
		count_1000 = 0
		count_1200 = 0
		count_1400 = 0
		for x in range(0,x_border):
			for y in range(0,ylim-1):
				for z in range(0,z_border):
					voxel = imgData[x,y,z,0]
					if( voxel > 1000 ): 
						if( voxel > 1200 ):
							if( voxel > 1400 ):  
								count_1400 += 1
							else:
								count_1200 += 1
						else:
							count_1000 += 1

		brainregions.append(count_1000);
		brainregions.append(count_1200);
		brainregions.append(count_1400);
		# front-lower
		count_1000 = 0
		count_1200 = 0
		count_1400 = 0
		for x in range(x_border,xlim-1):
			for y in range(0,ylim-1):
				for z in range(0,z_border):
					voxel = imgData[x,y,z,0]
					if( voxel > 1000 ): 
						if( voxel > 1200 ):
							if( voxel > 1400 ):  
								count_1400 += 1
							else:
								count_1200 += 1
						else:
							count_1000 += 1

		brainregions.append(count_1000);
		brainregions.append(count_1200);
		brainregions.append(count_1400);
		# back upper
		count_1000 = 0
		count_1200 = 0
		count_1400 = 0
		for x in range(0,x_border):
			for y in range(0,ylim-1):
				for z in range(z_border,zlim-1):
					voxel = imgData[x,y,z,0]
					if( voxel > 1000 ): 
						if( voxel > 1200 ):
							if( voxel > 1400 ):  
								count_1400 += 1
							else:
								count_1200 += 1
						else:
							count_1000 += 1

		brainregions.append(count_1000);
		brainregions.append(count_1200);
		brainregions.append(count_1400);
		# front upper
		count_1000 = 0
		count_1200 = 0
		count_1400 = 0
		for x in range(x_border,xlim-1):
			for y in range(0,ylim-1):
				for z in range(z_border,zlim-1):
					voxel = imgData[x,y,z,0]
					if( voxel > 1000 ): 
						if( voxel > 1200 ):
							if( voxel > 1400 ):  
								count_1400 += 1
							else:
								count_1200 += 1
						else:
							count_1000 += 1

		brainregions.append(count_1000);
		brainregions.append(count_1200);
		brainregions.append(count_1400);

		whitematter.append(brainregions)
		printProgress(i, n_samples)

	printProgress(n_samples, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(whitematter,output)
	output.close()
	print "Done"
	return whitematter


def extractImgNumber(imgPath):
	imgName = imgPath.split(os.sep)[-1]
	imgNum = int(imgName.split('_')[-1][:-4])
	return imgNum
