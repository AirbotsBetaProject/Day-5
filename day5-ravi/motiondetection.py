import cv2
import time
import imutils
import numpy as np
# Start streaming images from your computer camera
feed = cv2.VideoCapture(0) 

# Define the parameters needed for motion detection
alpha = 0.02 # Define weighting coefficient for running average
motion_thresh = 35 # Threshold for what difference in pixels  
running_avg = None # Initialize variable for running average

k = 31
while True:
    grab,current_frame = feed.read()
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    smooth_frame = cv2.GaussianBlur(gray_frame, (k, k), 0)

    # If this is the first frame, making running avg current frame
    if running_avg is None:
        running_avg = np.float32(smooth_frame) 

    # Find absolute difference between the current smoothed frame and the running average
    diff = cv2.absdiff(np.float32(smooth_frame), np.float32(running_avg))

    # Then add current frame to running average after
    cv2.accumulateWeighted(np.float32(smooth_frame), running_avg, alpha)

    # For all pixels with a difference > thresh, turn pixel to 255, otherwise 0
    _, subtracted = cv2.threshold(diff, motion_thresh, 1, cv2.THRESH_BINARY)
    cv2.imshow('Actual image', current_frame)
    cv2.imshow('Gray-scale', gray_frame)
    cv2.imshow('Smooth', smooth_frame)
    cv2.imshow('Difference', diff)
    cv2.imshow('Thresholded difference', subtracted)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cv2.destroyAllWindows()
feed.release()
