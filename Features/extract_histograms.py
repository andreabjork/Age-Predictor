import sys
import os
n_args = 2
help = "\n\nThis script requires "+str(n_args)+" arguments.\nThe first should be that maxHistogramValue and the second should be the directory containing the image files.\nExample: "+sys.argv[0]+" 4500 data/set_train\n\n"
if len(sys.argv) != n_args+1:
	print help
	exit()

import nibabel as nib
import glob
import pickle


hist_max_value = int(sys.argv[1])
imgPath = os.path.join("..",sys.argv[2],"*")
print "\nhist_max_value = "+str(hist_max_value)+"  "+str(type(hist_max_value))
print "imgPath = "+imgPath+"\n\n"

outputFileName = "histograms_"+str(hist_max_value)+"_"+sys.argv[2].replace(os.sep,"-")+".feature"
if os.path.isfile(outputFileName):
	print "Histograms matching the specifications already exist and are stored in:"
	print outputFileName
	exit()

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

output = open(outputFileName,"w")
pickle.dump(histograms,output)
output.close()
print "Done"