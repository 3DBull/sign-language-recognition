# Data Collection for Gesture Model
# The images will be collected in a folder and the corisponding labels will be
# stored in a np.array

import numpy as np
import cv2
import time

#Get video from webcam
#Image size 480x640
image_dir = 'CollectedImages'
label_dir = 'CollectedLabels'

dataNumber = np.load('dataNumber.npy') #Keeps a record of the number of datum for file naming 
video_capture = cv2.VideoCapture(2)

font = cv2.FONT_HERSHEY_SIMPLEX
classes = np.arange(ord('0'),ord('9')+1,1) #valid classes are numbers 0-9
np.append(classes,10)  #class 10 is used for a blank screen 

capTime = 0
capVal = 0

#Capture Frames from video feed
while True:
    ret, image = video_capture.read()
    key = cv2.waitKey(10)
    if key == ord('-'):  #Record blank when '-' key is pressed
        key = 10
    
    if key in classes:
        key = key-48  #This gets the number from keyboard unicode
        capVal = key
        key = np.array([key])
        
        #save image to images directory with data number and save label to label directory with data number
        cv2.imwrite(str(image_dir)+'/image'+str(dataNumber)+'.jpg', image)
        np.save(str(label_dir)+'/label'+str(dataNumber)+'.npy',key)
        dataNumber = dataNumber +1
        
        
        print('Recorded: '+str(key))
        capTime = 100
        
        
    if key == 27:
              break
    if capTime>0:
        cv2.putText(image, 'Captured a ' + str(capVal), (20, 30), font, .6, (0, 255, 0), 2, cv2.LINE_AA)
        capTime = capTime -1 	#Count down text display frames
    cv2.imshow('Capture',image)
    
print('Data Number: '+str(dataNumber))
np.save('dataNumber.npy',dataNumber)  #Store the data number so collection can pick up where it left off

video_capture.release()
cv2.destroyAllWindows()
