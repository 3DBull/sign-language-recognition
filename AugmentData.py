# Takes a dataset and augments it
# Returns only the augmented data

import numpy as np
import cv2
import random

def augmentData(data, h_flip=True, v_flip=False, zoom_range=0.2, rotation=20, h_shift=0.2, v_shift=0.2):
    images = []
    
    for im in data:

        #Apply h_flip
        if h_flip and (round(random.random())==1):
            im = cv2.flip(im,1)

        #Apply v_flip
        if v_flip and (round(random.random())==1):
            im = cv2.flip(im,0)

        #Apply Zoom
        p = int(random.uniform(100-zoom_range*100,100+zoom_range*100))
        im = zoom(im, int(p), (im.shape[0]//2, im.shape[1]//2))

        #Apply Rotation
        im = rotate(im, random.uniform(-rotation,rotation))

        #Apply Horizantal Shift
        im = hShift(im, int(random.uniform(-h_shift*im.shape[1],h_shift*im.shape[1])))

        #Apply Vertical Shift
        im = vShift(im, int(random.uniform(-v_shift*im.shape[0],v_shift*im.shape[0])))

        images.append(im)
    return np.array(images)
    

#Zoom on image given 'percentage' to zoom and 'center' as focal point of zoom
def zoom(im, percentage, center):
    im_shape = im.shape[:2]
	
	#Zoom in 
    if(percentage>100): 
        v_range = int(im_shape[0]*100/percentage/2) 
        h_range = int(im_shape[1]*100/percentage/2)
        roi = im[center[0]-v_range:center[0]+v_range,center[1]-h_range:center[1]+h_range,:]#get smaller region of interest
        im = cv2.resize(roi, (im_shape[1],im_shape[0])) #streatch it to original size
		
	#Zoom out
    else:               
        roi = cv2.resize(im, (int(im_shape[1]*percentage/100), int(im_shape[0]*percentage/100))) #make it smaller
        im = cv2.blur(im, (31,31))
        im[center[0]-roi.shape[0]//2:center[0]-roi.shape[0]//2+roi.shape[0], \
           center[1]-roi.shape[1]//2:center[1]-roi.shape[1]//2+roi.shape[1], :] = roi #place it on larger image
    
    return im


#Rotates image 'angle' degrees
def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


#Shifts image horizontally by 'shift' pixels
def hShift(im, shift): 
    if shift>0:
        roi = im[:, :im.shape[1]-shift, :]
        im = cv2.blur(im, (31,31))
        im[:, shift:, :] = roi
    else:
        roi = im[:, -shift:, :]
        im = cv2.blur(im, (31,31))
        im[:, :im.shape[1]+shift, :] = roi
    return im


#Shifts image vertically by 'shift' pixels
def vShift(im, shift):
    if shift>0:
        roi = im[:im.shape[0]-shift, :, :]
        im = cv2.blur(im, (31,31))
        im[shift:, :, :] = roi
    else:
        roi = im[-shift:, :, :]
        im = cv2.blur(im, (31,31))
        im[:im.shape[0]+shift, :, :] = roi
    return im
