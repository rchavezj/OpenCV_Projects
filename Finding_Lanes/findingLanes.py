import cv2
import numpy as np
 
# Canny edge detection 
# It's easier to find edges between pixels if 
# we're able to convert the enire image into gray. 
def canny(img):
    # Turn the images Gray
    # Processing a single channel instead of 
    # three (R/G/B) color image, is a lot more faster.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Smooth and reduce noise (Gaussian Filter)
    # To understand the concept of a Gaussian filter
    # recall that an image is stored as a collection 
    # of discrete pixels. Each of the pixels for a 
    # grayscale image is represented by a single number
    # that discribes the brightness of the pixel. For the 
    # sake of example, how do we smooth in the following 
    # image? We modify the value of a pixel with the average
    # value of the pixel intensities around it. Averaging out
    # out the pixels in the image to reduce noise will be 
    # done with the kernel (5x5). Each element within the
    # kernel contain gaussian values. 
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    # Gradient Intensity 
    # Identifying edges by seeing a sharp change in color
    # between adjacent pixels in the image. The change in
    # brightness over a series of pixels is the gradient. 
    # A strong gradient indicates a steep change wheras a
    # small gradient is a shallow change. We first establish
    # that an image as it's composed 
    # 50 lowest thershold
    # 150 highest thereshold
    canny = cv2.Canny(gray, 50, 150)
    return canny

# We need to mask the image to 
# a region where we need to detect
def region_of_interest(canny):
    # y-axis 
    # Bottom left corner of an image
    height = canny.shape[0]
    # x-axis
    # Bottom left corner of an image
    width = canny.shape[1]
    # Creates an array, size equal to
    # the image, of zero to make every
    # thing black outside the region
    # of interest. 
    mask = np.zeros_like(canny)
    # Region of 
    # interest boundaries
    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)
    # Fill the mask with the polygon 
    # (triangle) shape
    cv2.fillPoly(mask, triangle, 255)
    # Do a bitwise "AND" logic comparison 
    # with white and black pixels beteen the 
    # masked image (black background with white
    # triangle covering the lane) against the 
    # canny image (original image with edge 
    # detection).
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# (1) It's important to design a houghLines transform to 
# design the lines for our given traffic lane. 
# instead of using the cartesian coordinates (y and m):
# y = mx + b
# (2) We will be using the polar coordinates (row [p] and 
# theta): p = xcos(theta) + ysin(theta)
# (3) Arguments: 
# (3a) Image, number of pixels per bin, degree, 
# radians (np.pi/180 == 1 radians) 
# (3b) the minimum number of votes (intersection) in 
# hough-space for a bin needs to be 100 for it to be 
# accepted as a relevent line in describing our data 
# (3c) placeholder array to pass in content 
# (3d) length of a line in pixels we will accept into
# the output (minLineLength) 
# (3e) maximum distance in pixels between segmented 
# lines (maxLineGap) to be connected into one line. 
def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

# (1) Taking the sum of our color image 
# with our line image (black-blue lines)
# (2) Arguments:
# (2a) Lane image
# (2b) Taking the weighted sum between the arrays 
# between the two images provided in this function. 
# Will give a value of 0.8 that multiple each element
# within the first image provided in the input function.
# It will decrease the pixel intensity. 
# (2c) Taking the weighted sum between the arrays 
# between the two images provided in this function. 
# Will give a value of 1 that multiple each element
# within the second image provided in the input function.
# It will increase the pixel intensity compared to the first
# image. The lines will be more revealing than the frame. 
# (2d) Gamma argument where we can choose some values 
# that will add to our some. We put a scalue value of 1 which
# won't put sabstantial difference. 
def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
# This will return a black image
# except within the region of, 
# interest will have blue lines
# displayed within the traffic lane.
def display_lines(img,lines):
    # Start with a black image 
    # with the same dimensions as the
    # original picture of the road
    line_image = np.zeros_like(img)
    if lines is not None:
        # iterate existing 
        # line points given
        # from hough lines
        for line in lines:
            for x1, y1, x2, y2 in line:
                # (255,0,0): Blue color
                # Line thickness
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

# 
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
# 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

# Main program
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    cv2.imshow("result", combo_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

