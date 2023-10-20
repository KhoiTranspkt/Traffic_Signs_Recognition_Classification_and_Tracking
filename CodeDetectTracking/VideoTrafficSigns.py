import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
import os
from pathlib import Path
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
#Load video
filepath_video = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\FinalProject\VideoMorningAfternoon\Trafficsignsmorning.mp4'
cap = cv.VideoCapture(filepath_video)

#Load file cascade
filepath_cascade = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\FinalProject\CodeDetectTracking\TS2.xml'
trafficsigns_cascade = cv.CascadeClassifier(filepath_cascade)

#Input
isTracking = 0
count = 0
max_trackingFrame = 1

# Create Sequential Model
model = Sequential()

#Building the model
model = Sequential()
model.add(tf.keras.Input(shape = (30,30,3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))     

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(28, activation='softmax'))

filepath_model = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\FinalProject\CodeDetectTracking\my_model65.h5'
model= load_model(filepath_model)
model.load_weights(filepath_model)

#Assign labels
cv.ocl.setUseOpenCL(False)
signs_dictionaries=  {0:'Max Speed Limit (20km/h)',
                1:'Max Speed Limit (30km/h)',
                2:'Max Speed Limit (50km/h)',
                3:'Max Speed Limit (60km/h)',
                4:'Max Speed Limit (70km/h)',
                5:'Max Speed Limit (80km/h)',
                6:'Free Speed Limit (80km/h)',
                7:'Max Speed Limit (100km/h)',
                8:'Max Speed Limit (120km/h)',
                9:'No Passing',
                10:'No Passing For Truck',
                11:'Crossroad Ahead',
                12:'Priority Begin',
                13:'Priority',
                14:'Stop',
                15:'No Entry',
                16:'No Trucks',
                17:'One-way Traffic',
                18:'Danger',
                19:'No Turn Left',
                20:'No Turn Right',
                21:'Double Curve',
                22:'Bumpy Road',
                23:'Slippery Road',
                24:'Road Narrows on Right Sign',
                25:'Road Works',
                26:'Traffic Signals',
                27:'Pedestrians',
                28:'Children ',
                29:'Bicycles ',
                30:'Snow',
                31:'Animal',
                32:'End of Ban',
                33:'Turn Right Ahead',
                34:'Turn Left Ahead',
                35:'Ahead Only',
                36:'Go Straight or Right',
                37:'Go Straight or Left',
                38:'Keep Right',
                39:'Keep Left',
                40:'Roundabout ',
                41:'End of No Passing',
                42:'End of No Passing by Trucks'}
while True: 
    ret, frame = cap.read() # read a frame from video/camera
    w = frame.shape[1]
    h = frame.shape[0]
    if not ret:
        print('Can not receive frame or video ended')
        break
    center_points_currentframe = []
    # Detect traffic signs
    signs = trafficsigns_cascade.detectMultiScale(frame,1.3,5)
    if isTracking == 0:
        # Detection code
        rois=[]
        for (x, y, w, h) in signs:
            #crop roi
            roi = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 4)
            roi = cv.resize(roi,(30,30), interpolation=cv.INTER_AREA)
            #Draw center points
            cx = int((x + x + w )/ 2)
            cy = int((y + y + h)/2)
            center_points_currentframe.append((cx,cy))
            for p in center_points_currentframe:
                cv.circle(frame, p, 5, (0,0,255), -1)
            # predict
            signs = model.predict(np.array(roi).reshape(-1,30,30,3))
            maxindex = np.argmax(signs)
            cv.putText(frame, 'Class: '+ signs_dictionaries[maxindex],(x,y-int(w/8)),cv.FONT_HERSHEY_COMPLEX,0.4,(255,255,0),1)
            print(signs)
        # Create and initilize the tracker
        trackers = cv.legacy.MultiTracker_create()
        for roi in rois:
            trackers.add(cv.legacy.TrackerCSRT_create(), frame, roi)
        # Set istracking bit
        isTracking = 1
    else: 
        if count == max_trackingFrame:
            isTracking = 0
            count = 0
        # Update object location
        ret, objs = trackers.update(frame)
        if ret:
            for obj in objs:
                p1 = (int(obj[0]), int(obj[1]))
                p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
                cv.rectangle(frame, p1, p2, (255,0,0), 2)                
        else:
            print("Tracking fail")
            isTracking = 0
        count = count + 1
    cv.imshow('Final Result', frame) 
    #Close video 
    if cv.waitKey(1) == ord('q'): #Set the key for turning off video
        break
cap.release()
cv.destroyAllWindows()
