
import os
import nibabel as nib
import glob
import pickle

# imgDirRelPath must be a string and maxValue must be an integer
def extractHistograms(imgDirRelPath, maxValue = 4500):
	hist_max_value = int(maxValue)
	imgPath = os.path.join(imgDirRelPath,"*")
	print "\nhist_max_value = "+str(hist_max_value)+"  "+str(type(hist_max_value))
	print "imgPath = "+imgPath+"\n\n"

	outputFileName = "histograms_"+str(hist_max_value)+"_"+imgDirRelPath.replace(os.sep,"-")+".feature"
	if os.path.isfile(outputFileName):
		save = open(extractHistograms,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train
	allImageSrc = glob.glob(imgPath)

	histograms = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		print str((i*100)/n_samples)+"% working..."
		hist = [0]*hist_max_value
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		## for histogram
		for val in imgData.flatten().tolist():
			if val < hist_max_value:
				hist[val] += 1
		histograms.append(hist)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"
	return histograms