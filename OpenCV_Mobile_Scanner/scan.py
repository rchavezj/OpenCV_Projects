# import the necessary packages
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2
 
# the parser will be the image we'll be utilizing 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())