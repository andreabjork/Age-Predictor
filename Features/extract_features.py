import os
import inspect
import nibabel as nib
import numpy as np
import scipy as sp
import scipy.ndimage.interpolation as interpolation
import glob
import pickle
import math
from printProgress import printProgress

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# The directory to store the precomputed features
featuresDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# imgDirFullPath must be a string and maxValue must be an integer
# nPartitions is the "resolution" of the histogram
def extractHistograms(imgDirFullPath, maxValue = 4000, nPartitions = -1):
	if nPartitions == -1: nPartitions=maxValue
	hist_max_value = int(maxValue)

	# The number of different intensities per point of the histogram
	clusterSize = math.ceil((maxValue*1.)/nPartitions)
	imgPath = os.path.join(imgDirFullPath,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than neccesary
	outputFileName = os.path.join(featuresDir,"histograms_"+str(nPartitions)+"-"+str(hist_max_value)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*nPartitions
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		# Count occurances of each intensity below the maxValue
		for val in imgData.flatten().tolist():
			if val < hist_max_value:
				c = int(val/clusterSize)
				hist[c] += 1
		histograms.append(hist)
		printProgress(i+1, n_samples)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"

	return histograms

# This was an attempt at a more sophisticated feature using agglomerative clustering to define "colors"
# and then taking a histogram of those color. This did not prove to give better results.
def extractHierarchicalClusters(imgDirFullPath, n_clusters=10, ignoreCache=False):
	imgPath = os.path.join(imgDirFullPath,"*")

	outputFileName = os.path.join(featuresDir,"hierarchicalclusters_"+str(n_clusters)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName) and not ignoreCache:
		save = open(outputFileName,'rb')
		clusters = pickle.load(save)
		save.close()
		return clusters

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	clusters = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*n_clusters
		img = nib.load(allImageSrc[i])
		imgData_original = np.asarray(img.get_data()[:,:,:,0])
		# Resize to 10% of original size for faster processing
		imgData_resized = sp.ndimage.interpolation.zoom(imgData_original,0.10)
		imgData = np.reshape(imgData_resized,(-1,1))

		connectivity = grid_to_graph(*imgData_resized.shape)
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
		labels = ward.fit_predict(imgData).flatten().tolist()

		for lab in labels:
			hist[lab] += 1
		clusters.append(hist)
		printProgress(i+1, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(clusters,output)
	output.close()
	print "Done"
	return clusters

# The AgglomerativeClusters approach was suffering from the severely reduced "resolution" of the images
# and this was an attempt to improve on that by only looking at one slice of the image instead of reducing the
# "resolution". This too was unsuccessful.
def extractHierarchicalClustersSingleSlice(imgDirFullPath, n_clusters=10, ignoreCache=False):
	imgPath = os.path.join(imgDirFullPath,"*")

	outputFileName = os.path.join(featuresDir,"hierarchicalclusterssingleslice_"+str(n_clusters)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName) and not ignoreCache:
		save = open(outputFileName,'rb')
		clusters = pickle.load(save)
		save.close()
		return clusters

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	clusters = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*n_clusters
		total_intensities = [0]*n_clusters
		img = nib.load(allImageSrc[i])
		imgData_original = np.asarray(img.get_data()[:,:,:,0])
		brainSlice = imgData_original[:,:,imgData_original.shape[2]/2]
		imgData = np.reshape(brainSlice,(-1,1))

		connectivity = grid_to_graph(*brainSlice.shape)
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
		labels = ward.fit_predict(imgData).flatten().tolist()

		for j, lab in enumerate(labels):
			intensity = imgData[j][0]
			total_intensities[lab] += intensity
			hist[lab] += 1
		avg_intensity = np.asarray(total_intensities)*1./np.asarray(hist)
		avg_intensity = avg_intensity.flatten().tolist()
		avg_intensity, hist = zip(*sorted(zip(avg_intensity,hist)))

		clusters.append(hist)
		printProgress(i+1, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(clusters,output)
	output.close()
	print "Done"
	return clusters

def extractImgNumber(imgPath):
	imgName = imgPath.split(os.sep)[-1]
	imgNum = int(imgName.split('_')[-1][:-4])
	return imgNum
