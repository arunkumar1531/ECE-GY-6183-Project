#!usr/bin/env python

# Importing all the necessary libraries
# The integral library of this project is OpenCV
import cv2
import numpy as np
import os
import time
import glob

from collections import deque
from imutils.video import VideoStream

import imutils
import argparse

#These paths are used only when a input video is provided.
# creating new paths to create a folder to save the tracked frames.
wrk_dir = os.path.realpath(os.path.dirname(__file__))
new_dir = os.path.join(wrk_dir, "frames")

# The dictionary below includes all the available colors that can be detected by this python script.
# The HSV color space range of the colors are represented using tuples.
color_dict_HSV = {"black": [(180, 255, 30), (0, 0, 0)],
              "white": [(180, 18, 255), (0, 0, 231)],
              "red": [(180, 255, 255), (159, 50, 70)],
              "green": [(89, 255, 255), (36, 50, 70)],
              "blue": [(128, 255, 255), (90, 50, 70)],
              "yellow": [(35, 255, 255), (25, 50, 70)],
              "purple": [(158, 255, 255), (129, 50, 70)],
              "orange": [(24, 255, 255), (10, 50, 70)],
              "gray": [(180, 18, 230), (0, 0, 40)],
              "cyan": [(62, 255, 255), (29, 88, 10)]}

# Accepting a few inputs through command line arguments
# 1st argument - The object (ball) color to be tracked
# 2nd argument - This is the length of the trail that contains the previous locations of the object being tracked.
# 3rd argument - The path to the video file (we can use webcam as input or a video file.)
# If the 3rd argument is not given in CLI, then the webcam is used as input. 
# 4th argument - The thickness of the tracking path 
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--object_color", default = "blue", type=str, help="Color of the ball you want to track", required=False)
parser.add_argument("-t", "--trail_length", default=64, type=int, help="maximum size of the tracking trail", required=False)
parser.add_argument("-i", "--input_video", help="path/to/video/file.mp4", required=False)
parser.add_argument("-w", "--trail_width", default=5, type=float, help="Thickness of the trackig path - [0, 10]", required=False)
args = parser.parse_args()

# Removing any redundant spaces present in the command line input
# Converting to lowercase so that the command line input matches with the dictionary keys. 
args.object_color = args.object_color.replace(" ", "")
args.object_color = args.object_color.lower()

# creating folders with input video file as the name of the folder
if args.input_video:
	video_filename = os.path.basename(args.input_video).split(".")[0]
	new_dir = os.path.join(new_dir, video_filename)

# If the user provides a color that is not present in the dictionary, we have to raise an error and print the available color options.
if args.object_color not in color_dict_HSV.keys():
	print(f"Color not available. The available color options are {list(color_dict_HSV.keys())}")
	quit()

# Checking whether the provided video path exists. If it exists, We are checking if it is a video file or not.
# If not, appropriate error messages are displayed.
if args.input_video:
	if os.path.exists(args.input_video):
		if os.path.isfile(args.input_video):
			pass
		else: 
			print("The provided path is not a video file")
			quit()
	else: 
		print("Path does not exist")
		quit()


'''We are using deque to create an array to store the values of the tracked path.
The longer the array size, the longer the memory. Subsequently the path of the ball looks longer.
Deque is used instead of list because it is much faster that list for opeations like pop(), append(), etc.
Also the complexity of such methods is  O(1) in deque in contrast to O(N) in lists.'''

trail = deque(maxlen=args.trail_length)
# print(trail)

# Condition to decide whether to use the webcam or the video given through CLI.
if not args.input_video:
	# print("1")
	cap = cv2.VideoCapture(0) # Start the stream
else:
	# print("2")
	cap = cv2.VideoCapture(args.input_video)

# A buffer time of 1 second to load the webcam or input video file accordingly.
time.sleep(1)

# Initializing counter variable to number the frames while storing it. 
# Only used when a video file is given as input and not while using webcam input.
count = 1 

while True:

	# Starting to read frame by frame.
	out, frame = cap.read()
	if not out:
		break

	if not args.input_video:
		frame = cv2.flip(frame, 1)	
	# We have set the size of the display window as (1000, 750)
	frame = cv2.resize(frame, (1000,750))

	# In order to remove noise from the frames, the frames are being smoothed by gaussian filter of size (7,7)
	# We chose (7, 7) as a trade-ff between image quality and noise reduction.
	# We also tried using cv2.blur() - averaging filter. Gaussian filter produced good results.
	# blured_frame = cv2.blur(frame, (5,5))
	blured_frame = cv2.GaussianBlur(frame, (7, 7), 0)
	
	'''we can convert the frames from RGB color space to HSV color space.
	Object detection is much effective in HSV color space than RGB, so I convert RGB to HSV.
	However, the detection can be done in RGB color space as well.'''
	frame_HSV = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2HSV)
	# if not args.input_video:
	# 	frame_HSV = cv2.flip(frame_HSV, 1)	

	# mask1 = cv2.inRange(frame_HSV, (29, 88, 10), (62, 255, 255))
	# Initializing the mask based on the color input given by the user.
	# We perfom Morphological method (Opening - 2 iterations) which is a combination of eroding and dilation to accurately detect the target object. 
	mask1 = cv2.inRange(frame_HSV, color_dict_HSV[args.object_color][1], color_dict_HSV[args.object_color][0]) 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	open2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)	
	
	# Once the object is detected and asked, the contours are obtained.
	# We can use CHAIN_APPROX_NONE to save all the boundary points. However just a few points is enough to represent the contour.
	cntr = cv2.findContours(open2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	cntr = imutils.grab_contours(cntr)
	
	# Initializing the center of the bonding circle
	c = None 

	if len(cntr):
		max_cntr = max(cntr, key=cv2.contourArea)
		
		# Here we define the coordinates and the radius of the bounding circle.
		((x, y), radius) = cv2.minEnclosingCircle(max_cntr)
		
		# Here, we define the moment of the Image.
		# Moment of the image is usually used to find the centroid, area, radius, etc of the Image.
		# In this case we use it to find the centroid of the image frame.
		M = cv2.moments(max_cntr)
		c = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

		x = int(x)
		y = int(y)
		radius = int(radius)
		
		# For the object to be considered, the radius has to be atleast 10.
		# This can be modified depending on the size of the object being tracked.
		# Two different circles are defined 
		# one circle bounding the target object
		# Another circle denoting the center of the bounding circle.
		if radius > 10:
			cv2.circle(frame, (x, y), radius, (255, 255, 0), 2)
			cv2.circle(frame, c, 5, (0, 0, 255), -1)

	## The tracked coordinates of the object are added to the deque.
	trail.appendleft(c)

	for i in range(1, len(trail)):
		
		# Initially there will not be any obect to track, in that case there will not be any tracking points.
		# In that case, it can neglected.
		# Or if the object is removed from the frame.
		if (trail[i - 1] or trail[i]) is None:
			continue
		
		# we define the thickness of the trail using the user input.
		# However, the default value is 5
		t = int(np.sqrt(args.trail_length / float(i + 1)) * args.trail_width)

		# We are constructing a line to follow the path of the object with the specified thickness.
		cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), t)

	# finally the frame is displayed for the user.
	# if the webcam is used as input, we directly display the frames.
	# when an input videois provided, the tracked video frames are stored in a folder "frames" created in the working directory.
	if not args.input_video:
		cv2.imshow("Frame", frame)
	else:
		cv2.imshow("Frame", frame)
		
		# checking if the "frames" directory is already present
		if os.path.isdir(new_dir):

			# adding the tracked video frames to the folder.
			file_name = f"frame_{count}.jpg"
			count+=1
			cv2.imwrite(os.path.join(new_dir, file_name), frame)
		else:

			# if the "frames" folder is not present, we create one and save the tracked video frames in it.
			os.makedirs(new_dir)
			file_name = f"frame_{count}.jpg"
			count+=1
			cv2.imwrite(os.path.join(new_dir, file_name), frame)

	# The window can be closed by pressing [Esc] key if input is via webcam (Real-time)
	# If the video is pre-loaded via Command line, the tracking window will automatically close once the video ends.
	k = cv2.waitKey(1) & 0xFF 
	if k == 27:
		break

# The code from line 212 to 232 is used to convert the tracked video frames to a mp4 video file.
# The mp4 video file is also sored in the same location as the tracked video frames.
if args.input_video:	
	img_array = []
	img_files = glob.glob(os.path.join(new_dir, '*.jpg'))

	num = 1
	for i in img_files:
	    for j in img_files:
	        if num<=len(img_files):
	            if int(os.path.basename(j).split("_")[1].split(".")[0]) == num:
	                img = cv2.imread(j)
	                # cv2.imwrite(path_, img)
	                height, width, layers = img.shape
	                size = (width,height)
	                img_array.append(img)
	                num+=1

	video_path = os.path.join(new_dir, "01_tracking_video.mp4")
	out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
	 
	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()

# finally, we release the frames and quit the windows opened by this script and close the program.
cap.release()
cv2.destroyAllWindows()
