#Importing
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import PIL
from PIL import Image
import skimage.io as io
from skimage.filters import rank

#Loading Images
di1 = 'C:\\Projects\\Images\\diestrus1.tif'
di2 = 'C:\\Projects\\Images\\diestrus2.tif'
es1 = 'C:\\Projects\\Images\\estrus.tif'
es2 = 'C:\\Projects\\Images\\estrus2tif'
met1 = 'C:\\Projects\\Images\\metestrus1.tif'
met2 = 'C:\\Projects\\Images\\metestrus2.tif'
pro1 = 'C:\\Projects\\Images\\proestrus1.tif'
pro2 = 'C:\\Projects\\Images\\proestrus2.tif'
ex1 = 'C:\\Projects\\Images\\example_cells_1.tif'
ex2 = 'C:\\Projects\\Images\\example_cells_2.tif'
img1 = io.imread(di1)
#Gaussian Smoothing
smooth = img1
sigma = 2
green = img1[:,:,2]
smooth1_smooth = ndi.filters.gaussian_filter(green,sigma)
plt.imshow(smooth1_smooth)
plt.show(smooth1_smooth.any())
#Adaptive background
struct = ((np.mgrid[:31,:31][0] - 15)**2 + (np.mgrid[:31,:31][1] - 15)**2) <= 15**2 
bg = rank.mean(smooth1_smooth, selem=struct)
#Threshold
threshold = smooth1_smooth >= bg
threshold = ndi.binary_fill_holes(np.logical_not(threshold))
plt.imshow(threshold,interpolation='none',cmap='gray')
plt.show(threshold.any())




plt.imshow(img1)
plt.show(img1.any())

green = img1[:,:,1]
smooth1_smooth = ndi.filters.gaussian_filter(green,sigma)
plt.imshow(smooth1_smooth)
plt.show(smooth1_smooth.any())

green = img1[:,:,2]
smooth1_smooth = ndi.filters.gaussian_filter(green,sigma)
plt.imshow(smooth1_smooth)
plt.show(smooth1_smooth.any())

green = img1[:,:,0]
smooth1_smooth = ndi.filters.gaussian_filter(green,sigma)
plt.imshow(smooth1_smooth)
plt.show(smooth1_smooth.any())
