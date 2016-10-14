import nibabel as nib
import sys

imgNumber = str(sys.argv[1])
image = nib.load('data/set_train/train_'+imgNumber+'.nii')
epi_img_data = image.get_data()
imgShape = epi_img_data.shape
imgHalfX = imgShape[0]/2
imgHalfY = imgShape[1]/2
imgHalfZ = imgShape[2]/2

import matplotlib.pyplot as plt
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[imgHalfX, 75:153, 74:107, 0]
slice_1 = epi_img_data[65:109, imgHalfY, 55:110, 0]
slice_2 = epi_img_data[55:120, 65:150, imgHalfZ, 0]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  
plt.show()

