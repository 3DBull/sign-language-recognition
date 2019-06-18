# Generates and trains a CNN model
# With augmentation enabled:
#   The program continues to train until the model reaches a threshold of accuracy
#   The model is saved at each epoch if it has a better accuracy than the last epoch
#   The training set is randomly augmented at each epoch to make the model more robust
#
# Without augmentation enabled:
#   The program simply trains with the given dataset for the gven number of epochs
#   The dataset can be doubled by mirroring each image (enable mirroring)


import numpy as np
import pickle
import os
import cv2
import AugmentData as dat
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, UpSampling2D, Flatten
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import losses, optimizers
from keras.utils import to_categorical

#Input: 480x640 image
#Output: number 0-9

####RunOptions####
augmentation = True
trainExisting = True
enableMirroring = True
modelName = 'displayModel_small_deep_aug.h5'
image_dir = 'CollectedImages/'
label_dir = 'CollectedLabels/'
acc_thresh = 0.99
batch_size = 50
epochs     = 40
##################

input_shape = (180, 240, 3)
num_classes = 11

model = Sequential()

#report_tensor_allocations_upon_oom

#BUILD THE MODEL LAYERS
### The Layers ###

model.add(Conv2D(8, kernel_size=(3, 3), strides=(2, 2),
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.1))
    
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))
### End Network ###


#COMPILE THE MODEL
model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=0.001), decay=0.001,
              metrics=['accuracy'])
model.summary()

#IMPORT THE TRAINING DATA
print('Collecting Training data...')
images =[]
labels = []

for imageName in os.listdir(image_dir):
    if imageName != 'desktop.ini':

        #get each image and resize it to the model input size
        im = cv2.resize(cv2.imread(image_dir+imageName) ,(input_shape[1],input_shape[0]))
        images.append(im)
        if enableMirroring:
            images.append(cv2.flip(im,1)) #Also add the flipped image to the data set

        #get each label corrisponding to the image
        labels.append(np.load(label_dir+'label'+imageName[5:-4]+'.npy'))
        if enableMirroring:
            labels.append(np.load(label_dir+'label'+imageName[5:-4]+'.npy'))

print('Length of dataset: '+str(len(labels)))

#TRAIN THE MODEL
print('Training...')
if trainExisting:
    model = load_model(modelName)


x_train, x_val, y_train, y_val = train_test_split(np.array(images), to_categorical(labels),
                                                  test_size=0.2,random_state=42)

if augmentation:

    prev_acc = 0
    epoch = 1
    learning_rate = 0.01
    stuckness = 0
    while prev_acc < acc_thresh:
        print()
        print('Epoch: '+str(epoch))
        if trainExisting:
            model = load_model(modelName)
            model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=learning_rate), metrics=['accuracy'])
            
        history = model.fit(dat.augmentData(x_train, h_flip=False), y_train, batch_size = batch_size, epochs = 1,
              verbose=0, validation_data=(x_val,y_val), shuffle=True)

        #get the accuracy information
        acc = history.history['acc'][-1]        
        val_acc = history.history['val_acc'][-1]
        print('acc: '+str(acc)+'   val_acc: '+str(val_acc))
        
        if val_acc > prev_acc:
            model.save(modelName)
            prev_acc = val_acc
            print('Saved Model')
            
            if val_acc > 0.8 and learning_rate>0.005:
                learning_rate = 0.005
                print('Learning Rate Decreased to 0.005')
            if val_acc > 0.9 and learning_rate>0.001:
                learning_rate = 0.001
                print('Learning Rate Decreased to 0.001')
            if val_acc > 0.99 and learning_rate>0.0001:
                learning_rate = 0.0001
                print('Learning Rate Decreased to 0.0001')
              

            stuckness = 0
                
            
        else:
            print("This round wasn't good enough to keep")
            trainExisting = True
            stuckness = stuckness + 1
            
            if stuckness > 5:
                learning_rate = learning_rate*3/4
                print('Learning Rate Decreased to '+str(learning_rate))
                stuckness = 0 
            
            
        epoch = epoch + 1

        if cv2.waitKey(1)==27: # Allow manual exit with the escape key
            break;
        
    
else:
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
              verbose=2, validation_data=(x_val,y_val), shuffle=True)

#EVALUATE THE MODEL
print('Evaluating')

scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                         
#SAVE THE MODEL
model.save(modelName)

