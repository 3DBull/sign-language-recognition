# This program captures frames from a webcam and 
# predicts hand signals in the number range 0-9.
# Predictions are made based on an externally trained 
# CNN that is loaded in.
# The program is specifically made to run on a raspberry pi. 
# It communicates with GPIO pins for direct power-up and 
# power-down functionality.  
# For resource protection, the program has a sleep mode 
# that it enters during periods of inactivity. 

record_video = False

#Load libraries
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import sys
import time
from threading import Thread
import RPi.GPIO as GPIO
import subprocess

#Load the CNN Model
model = load_model('displayModel_small_deep_aug.h5')
#input_shape = (240,320,3)
input_shape = (180, 240, 3)
screen_size = (720, 1280, 3)

#Get video from webcam
video_capture = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
# Uncomment to record a video clip of prediction

'''
if record_video:
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    size = width,height
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    out = cv2.VideoWriter("vid.avi", fourcc, 20.0, (screen_size[1],screen_size[0]))
'''


#Define Other things
font = cv2.FONT_HERSHEY_COMPLEX
MOTION_THRESH  = 1000000
MOTION_TIME    = 100
motion_timer = MOTION_TIME
exit_flag = False
prediction = 0
classes = np.arange(ord('0'),ord('9')+1,1)
np.append(classes, 10)

#Set-up full screen window and display splash screen
cv2.destroyAllWindows()
cv2.namedWindow('Hand Recognition', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hand Recognition',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#Initialize the screen buffer
screen = np.zeros(screen_size, dtype='uint8')
#Apply artwork to screen
graphic = cv2.imread('NetInv.png')
screen[:480,:320,:] = graphic  
cv2.line(screen,(0,screen_size[0]*2//3),(screen_size[1]//4,screen_size[0]*2//3),(0,255,0),thickness=3)
cv2.line(screen,(screen_size[1]//4,screen_size[0]*2//3),(screen_size[1]//4,screen_size[0]),(0,255,0),thickness=6)



#Compares the last two frames of video to check for motion
def checkMotion(prev_frame, frame):
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21,21),0)
    frame = cv2.GaussianBlur(frame, (21,21),0)
    diff = cv2.absdiff(prev_frame, frame)
    motion = np.sum(diff)
    #print(motion)
    return motion

#Start a thread to use power button interrupt
def setPowerInterrupt():
    global exit_flag
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.wait_for_edge(3, GPIO.FALLING)

    exit_flag = True

powerThread = Thread(target = setPowerInterrupt)
powerThread.start()

    
###MAIN LOOP###  
while not exit_flag:
    #get image from camera 
    ret, image = video_capture.read()

    #Get keyboard input
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    #Predict From Image
    image_small = cv2.resize(image, (input_shape[1], input_shape[0]))
    im = np.array(image_small)
    im = im[None,:,:,:]
    prediction = model.predict(im, batch_size = None, verbose = 0)
    p = np.argmax(prediction)
    
    #Place image in screen buffer    
    image = cv2.resize(image, (image.shape[1]*screen_size[0]//image.shape[0],screen_size[0]))
    screen[-input_shape[0]*4:,-input_shape[1]*4:,:] = image
    screen[screen_size[0]*2//3+2:-1,0:screen_size[1]//4-3,:] = 40   #Clear the Display Number area

    #If the prediction is a number, display it 
    if p != 10:
        cv2.putText(screen, str(p), (60, 706), font, 10, (255, 255, 250), 15, cv2.LINE_AA)
        motion_timer = MOTION_TIME

    #If the prediction is blank 
    else:
        motion_timer = motion_timer - 1

        #SLEEP ROUTINE
        while motion_timer <= 0:
            prev_im = image
            ret, image = video_capture.read()
            image = cv2.resize(image, (image.shape[1]*screen_size[0]//image.shape[0],screen_size[0]))
            screen[-input_shape[0]*4:,-input_shape[1]*4:,:] = image
            cv2.imshow('Hand Recognition', screen)

            #Check for motion. If motion is detected, add time to the timer and go back to main loop
            if checkMotion(prev_im, image)>MOTION_THRESH: 
                motion_timer = MOTION_TIME
            else:
                cv2.waitKey(10)

            if exit_flag:
                break        
            
    

    #Refresh the screen 
    cv2.imshow('Hand Recognition', screen)
    '''
    if record_video:
        out.write(screen) # Write out frame to video
    #print(prediction)
    '''

video_capture.release()

if record_video:
    out.release()

cv2.destroyAllWindows()

if exit_flag:
    subprocess.call(['shutdown', '-h', 'now'], shell=False) #Shutdown the PI when done
