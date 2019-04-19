
<img src="https://media.giphy.com/media/vwFITsRSS3cbgFNuaY/giphy.gif" width="900" height="500" />

```python
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
```
![alt text](https://github.com/rchavezj/OpenCV_Projects/blob/master/Finding_Lanes/images/cannyEdgeDetection.png)

```python
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
```
![alt text](https://github.com/rchavezj/OpenCV_Projects/blob/master/Finding_Lanes/images/regionOfInterest.png)


```python
def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)
        
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*2.5/5)      # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
# Our current lines are noisy from  
# the hough transform so we need to
# average out the lines to smooth it out.
def average_slope_intercept(image, lines):
    # Keeping an array to
    # track all positions.
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        # These are the points for a line
        for x1, y1, x2, y2 in line:
            # polyfit returns the slope (m) and
            # intercept (b) for y = mx + b
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            # Remember we have two slopes 
            # from the traffic lane. If the slope
            # from the two is negative (rise/run), 
            # we're currently looking at the left lane. 
            # Otherwise we're looking at the right lane.
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                # Positive slope. 
                right_fit.append((slope, intercept))
    # Return smooth slopes
    # Using numpy functions to find the 
    # average within the left and right slope. 
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # Place the new smooth slope on the new image.
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    # Return an array of the two new 
    # lines placed on the image/frame. 
    averaged_lines = [left_line, right_line]
    return averaged_lines
```
![alt text](https://github.com/rchavezj/OpenCV_Projects/blob/master/Finding_Lanes/images/averageSmoothLines.png)


```python
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
```
![alt text](https://github.com/rchavezj/OpenCV_Projects/blob/master/Finding_Lanes/images/findingLanes.png)





