#Gesture Model
import numpy as np
import pickle
import os
import cv2
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
augmentation = False
trainExisting = True
modelName = 'displayModel_N.h5'
image_dir = 'CollectedImages/'
label_dir = 'CollectedLabels/'

batch_size = 50
epochs     = 20
#training_length = 1000
input_shape = (240,320,3)
num_classes = 10
model = Sequential()

#report_tensor_allocations_upon_oom

#BUILD THE MODEL LAYERS
### The Layers ###

model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
                 input_shape=input_shape))
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
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
model.summary()

#COLLECT THE TRAINING DATA
print('Collecting Training data...')
images =[]
labels = []

for imageName in os.listdir(image_dir):
    if imageName != 'desktop.ini':

        #get each image and resize it to the model input size
        im = cv2.resize(cv2.imread(image_dir+imageName) ,(input_shape[1],input_shape[0]))
        images.append(im)
        images.append(cv2.flip(im,1))

        #get each label corrisponding to the image
        labels.append(np.load(label_dir+'label'+imageName[5:-4]+'.npy'))
        labels.append(np.load(label_dir+'label'+imageName[5:-4]+'.npy'))

print('Length of dataset: '+str(len(labels)))

#TRAIN THE MODEL
print('Training')
if trainExisting:
    model = load_model(modelName)

x_train, x_val, y_train, y_val = train_test_split(np.array(images), to_categorical(labels),
                                                  test_size=0.2,random_state=42)

if augmentation:
    #create a datagenerator that will augment the data while training
    datagen = ImageDataGenerator( 
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(x_train)
    #fits the model on batches with data augmentation 
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(images)/batch_size,epochs=epochs,
                        verbose=2, validation_data=(x_val,y_val), shuffle=True)
    
else:
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
              verbose=2, validation_data=(x_val,y_val), shuffle=True)

#EVALUATE THE MODEL
print('Evaluating')

scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                         
#SAVE THE MODEL
model.save(modelName)

