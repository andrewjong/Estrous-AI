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
img = io.imread(di1)
#Gaussian Smoothing
sigma = 2
smooth = img[:,:,2]
smooth_filter = ndi.filters.gaussian_filter(smooth,sigma)
plt.imshow(smooth_filter)
plt.show(smooth_filter.any())
#Adaptive background
struct = ((np.mgrid[:31,:31][0] - 15)**2 + (np.mgrid[:31,:31][1] - 15)**2) <= 15**2 
bg = rank.mean(smooth_filter, selem=struct)
#Threshold
threshold = smooth_filter >= bg
threshold = ndi.binary_fill_holes(np.logical_not(threshold))
plt.imshow(threshold,interpolation='none',cmap='gray')
plt.show(threshold.any())