import numpy as np
import cv2

feed = cv2.VideoCapture(0)

alpha = 0.02 
motion_thresh = 35 
running_avg = None

k = 31
while True:
    current_frame = feed.read()[1]
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    smooth_frame = cv2.GaussianBlur(gray_frame, (k, k), 0)
    
    if running_avg is None:
        running_avg = np.float32(smooth_frame)

    diff = cv2.absdiff(np.float32(smooth_frame), np.float32(running_avg))

    cv2.accumulateWeighted(np.float32(smooth_frame), running_avg, alpha)

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
