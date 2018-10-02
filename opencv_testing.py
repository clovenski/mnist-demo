#!/usr/bin/python

from math import ceil
import numpy as np
import cv2 as cv
import tensorflow as tf
from scipy import ndimage

model = tf.keras.models.load_model('model.h5')

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 1.0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, data_raw = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)

        if np.sum(data_raw) == 0:
            continue

        roi_x1 = roi_y1 = 0
        roi_y2 = gray.shape[0] - 1
        roi_x2 = gray.shape[1] - 1

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

        # clearer region of interest
        roi_x1 -= 5
        roi_y1 -= 5
        roi_x2 += 5
        roi_y2 += 5
        # build region of interest vertices
        roi_pt1 = (roi_x1, roi_y1)
        roi_pt2 = (roi_x2, roi_y2)

        data = cv.resize(data_raw, (8, 8), interpolation=cv.INTER_AREA)
        data = cv.resize(data, (20, 20), interpolation=cv.INTER_AREA)
        data = data / 255.0

        data = np.pad(data, 4, mode='constant', constant_values=0.0)

        com = ndimage.measurements.center_of_mass(data)
        y_offset = ceil(13.5 - com[0])
        x_offset = ceil(13.5 - com[1])
        M = np.array([[1,0,x_offset],
                      [0,1,y_offset]])
        
        data = cv.warpAffine(data, M, (28, 28))

        prediction = np.argmax(model.predict(data[np.newaxis,:,:,np.newaxis]))

        cv.rectangle(gray, roi_pt1, roi_pt2, 0)
        cv.putText(gray, str(prediction), (0, gray.shape[0]-10), cv.FONT_HERSHEY_DUPLEX, 5, 0, 5)

        cv.imshow('Webcam Feed', gray)
        cv.imshow('Input to Model', cv.resize(data, dsize=(0,0), fx=10, fy=10, interpolation=cv.INTER_AREA))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()