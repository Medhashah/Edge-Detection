from CannyEdgeDetection import canny_edge_detector
import cv2
import sys

try:
	e = [] #empty edges.

	Input_image = str(sys.argv[1]) #getting image path
	image = cv2.imread(Input_image)	# read image
	gray_level_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert into grayscale image.

	canny_edge_detector(Input_image, gray_level_image, e) #getting edges

except:
	print("Please provide image path")


