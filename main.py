#!/usr/bin/python

from __future__ import print_function
import sys
from math import ceil
import numpy as np
import cv2 as cv
import tensorflow as tf
from scipy import ndimage

dev_mode = False
if len(sys.argv) > 1 and sys.argv[1] == '-d':
    dev_mode = True

# load model saved by model_training.py
model = tf.keras.models.load_model('model.h5')

# pass with arg 1 for second video capturing device: usb webcam
cap = cv.VideoCapture(1)
# lower the framerate; although it doesn't seem to actually become 1 FPS
cap.set(cv.CAP_PROP_FPS, 1.0)

# while webcam source is still open
while cap.isOpened():
    # read a frame
    ret, frame = cap.read()
    # ret probably indicates frame was properly read; check opencv docs
    if ret == True:
        # grayscale the image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # binary inverted threshold, to get black background and white for digit;
        # 70 is used for threshold, adjust to indicate how dark something needs
        #   to be to appear white in data_raw
        _, data_raw = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)

        # if nothing in webcam frame is dark, data_raw is all black
        # need to continue loop to combat what the trimming part below does
        if np.sum(data_raw) == 0:
            continue

        # set up region of interest starting points;
        # top left of frame
        roi_x1 = roi_y1 = 0
        # bottom right of frame
        roi_y2 = gray.shape[0] - 1
        roi_x2 = gray.shape[1] - 1

        # Trim data_raw of any completely black outer rows and columns
        # until each side reaches the digit; essentially focus onto the digit,
        # setting region of interest as well

        # trim from top
        while np.sum(data_raw[0]) == 0:
            data_raw = data_raw[1:]
            roi_y1 += 1
        # trim from left
        while np.sum(data_raw[:,0]) == 0:
            data_raw = data_raw[:,1:]
            roi_x1 += 1
        # trim from bottom
        while np.sum(data_raw[-1]) == 0:
            data_raw = data_raw[:-1]
            roi_y2 -= 1
        # trim from right
        while np.sum(data_raw[:,-1]) == 0:
            data_raw = data_raw[:,:-1]
            roi_x2 -= 1

        # clearer region of interest; expand it a bit
        roi_x1 -= 5
        roi_y1 -= 5
        roi_x2 += 5
        roi_y2 += 5
        # build region of interest vertices
        roi_pt1 = (roi_x1, roi_y1)
        roi_pt2 = (roi_x2, roi_y2)
        # apply roi to webcam frame
        cv.rectangle(frame, roi_pt1, roi_pt2, 0)

        # get the dimensions of the digit (dim. of trimmed data)
        digit_shape = (roi_x2 - roi_x1 - 10, roi_y2 - roi_y1 - 10)
        if dev_mode:
            print('digit_shape:', digit_shape)
        # calculate how much to pad horizontally to make square shape
        x_pad = max(0, digit_shape[1] - digit_shape[0]) // 2
        # pad to make digit fit into a square shape
        data = np.pad(data_raw, ((0,), (x_pad,)), mode='constant', constant_values=0.0)
        # resize to 20x20
        data = cv.resize(data, (20, 20), interpolation=cv.INTER_AREA)

        # pad the 4 outer rows/columns of data to get 28x28 image
        data = np.pad(data, 4, mode='constant', constant_values=0.0)

        # get its center of mass
        com = ndimage.measurements.center_of_mass(data)
        # calculate its offset from center of image (13.5, 13.5)
        y_offset = ceil(13.5 - com[0])
        x_offset = ceil(13.5 - com[1])
        # create transformation matrix for warpAffine function of cv
        M = np.array([[1,0,x_offset],
                      [0,1,y_offset]])

        # shift the image so that center of mass is at the center of image (14, 14)
        data = cv.warpAffine(data, M, (28, 28))
        
        # normalize image to conform to what model is trained with
        data = data / 255.0
        # get model's prediction of the data
        prediction = np.argmax(model.predict(data[np.newaxis,:,:,np.newaxis]))

        # put this prediction on the webcam feed image for display
        cv.putText(frame, str(prediction), (0, gray.shape[0]-10), cv.FONT_HERSHEY_DUPLEX, 5, 0, 5)

        # show first window, showing webcam feed, along with prediciton and roi box
        cv.imshow('Webcam Feed', frame)
        # show second window, showing what the machine takes as an input
        cv.imshow('Input to Model', cv.resize(data, dsize=(0,0), fx=10, fy=10, interpolation=cv.INTER_AREA))

        # during runtime, hit the 'q' key to quit the program
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# destroy the windows created by cv
cap.release()
cv.destroyAllWindows()