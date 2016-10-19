import nibabel as nib


# For 'elementnumber' .nii files in path+'set_train', extract average intensity values
def extract_average(path,elementnumber): #filename,directory):
	# If the feature has already been calculated for all samples, read it from file.
	with open('feature_avg') as f:
		average = [float(x.strip('\n')) for x in f.readlines()]
	if(len(average) == elementnumber):
		print "Loaded feature: 'average intensity' from file 'feature_avg'."
		return average

	# Else calculate it (Warning: This may take up to 20 minutes.)
	else:
		print "File 'feature_avg' not found. Calculating averages intensity values."
		average = [None]*elementnumber;
		for filename in os.listdir("./data/set_train/"):
			if filename.endswith(".nii"):
				testnumber = int(filename.split('_')[1].split('.')[0])
				average[testnumber-1] = average(filename,"./data/set_train/")
				print "extracted filename: " + str(filename) + " w. number " + str(testnumber)

	# Save result to file.
	avg_file = open('feature_avg','w')
	for line in average:
		avg_file.write(str(line) + "\n")
	avg_file.close()
	print "Saved to file 'feature_avg'."

def average(filename,directory):
	img = nib.load(directory+filename)
	data = img.get_data()
	[xlim, ylim, zlim, Ilim] = data.shape
	#print filename +  " has shape " + str(data.shape)
	sum = 0.0
	for x in range(xlim):
		for y in range(ylim):
			for z in range(zlim):
				sum += data[x,y,z,0]
				#print str(x) + ' ' + str(y) + " " + str(z)
	avg = sum/((xlim+1)*(ylim+1)*(zlim+1))
	return avg

