import os
import glob
import time
import sys
import numpy as np
import imageio
import scipy

for file in glob.glob('Masks/*png') : 
	print 'Processing ', file
	im = imageio.imread(file)
	h, w, _ = im.shape
	offsetH = (w - h) / 2
	arr = np.zeros((w, w))

	arr[offsetH:offsetH+h,:] = im[:,:,0]

	imageio.imwrite(file, arr)
