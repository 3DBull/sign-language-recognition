# sign-language-recognition
Basic sign language recognition using a CNN to classify numbers 0-9
This program is installed on a raspberry pi and featured in the science building at LeTourneau University. 

![sign-language-in-action](https://github.com/3DBull/sign-language-recognition/blob/master/sign-language.jpg)


## Installation
Extract the 'CollectedImages' folder and the 'CollectedLabels' folder to the root directory. 

## Description of Files
### SignLanguageRecognition
Main prediction program designed to run on a raspberry pi 3. 

### ModelGenerator 
Sets up the structure for the neural network and trains it given the training data in 'CollectedImages' and 'CollectedLabels'

### DataCollect
Collects training data given the webcam and user input.

### AugmentData
Library created to augment data during model training. 
