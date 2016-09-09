# How To Build a Kick-Ass Mobile Document Scanner in Just 5 Minutes

OpenCV and Python versions:
This example will run on Python 2.7 and OpenCV 2.4.X.

In this lesson, we'll be making our very own Mobile Document Scanner! 

![alt tag](https://github.com/OverRatedTech/OpenCV_Projects/blob/master/OpenCV_Mobile_Scanner/receipt-scanned.jpg)

Theres going to be 3 simple steps in order for us to complete the app

	Step 1: Detect edges.

	Step 2: Use the edges in the image to find the contour (outline) representing the piece of paper being 
	scanned.

	Step 3: Apply a perspective transform to obtain the top-down view of the document.


First we need to first import a four_point_transform which is a " bird eye view" of an image with 
provided reference points (AKA "4 point perspective transform") 

Have a file "transform.py" to initialize a list (array) containing all of the coordinates. For our 
sake, we will be naming our list "rect" since four points corresponds to a square. 
```python
# import the necessary packages
import numpy as np
import cv2

# pts, which is the passing argument into this function, 
# contains the list of the four points specifying the (x,y)
# coordinates which defines a square/rectangle
def order_points(pts):
	# (1) initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	# (2) This line also requires allocation
	rect = np.zeros((4, 2), dtype = "float32")

	# The top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]	#top-left		mallest x + y sum
	rect[2] = pts[np.argmax(s)]	#bottom-right 	largest x + y sum

	# Now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]	#top-right
	rect[3] = pts[np.argmax(diff)]	#bottom-left

# return the ordered coordinates
return rect
```

There are two things we need to initially do 
in order to import packages : 

	(1) numpy 	--> numerical processing 
	(2) cv2 	--> OpenCV bindings

OpenCV Bindings: Enable users to code in python from a c++ framework in opencv. 

From the code above us, "Its important" to be consistant what points you
assign from the transformation view. The code below us will utilize the 
list we returned (rect = orderpoins(pts)), and store each element into an 
entry for top-left, top-right, bottom-left, bottom-right. 
```python
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
```
The whole point finding the maxWidth and maxHeight is to construct a 
destination view of a birds eye view. 


	















	
