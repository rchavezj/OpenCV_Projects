import cv2
import numpy as np

image = cv2.imread('test_img.jpg')
lane_image = np.copy(image)