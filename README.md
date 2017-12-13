[Youtube Link](https://youtu.be/wWIiQYm5TZI)

## Project Description
In this project I used convolutional neural networks to clone the driving behavior. I have used the sample data provided. 
The images are taken from diffrent angles left, center, right.
The network is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

### Files Included
model.py
model-000.h5
run1.mp4

The code works and is well commented. I have used generator with yield return due to memory limitations on the laptop.

### Model Architecture
I have added both the left and right images with a steering correction of 0.2 along with center images
I have used normalization and randon flip for preprosseing
I have used cropping to remove the unnecessary data from the image.

In the end, the model looks like as follows:

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
```

#### Attempts to reduce overfitting
I haven't used Dropout or pooling. I have split my traning and validations in the ratio of 80:20. I have used only 3 epochs

#### Model parameter tuning
Adam Optimizer. No learning rate provided
#### Appropriate training data
I have used the training data provided by udacity.

### Architecture and training documentation
1. Solution design approach
    My initial thought after looking at the data (8k+) input data was it was not sufficeint to train the model. Thinking in the lines how we can add more data to the training model gave me the idea to add both the left and right images to the traning model. As mentioned in the lesson nVidia model is good.
I used  [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) The only modification was to add a new layer at the end to have a single output as it was required.
Data Augmentation
    1. I cropped the image to remove the unneccesary data from the image.
    2. Randomly flipped the image
Increasing the epochs didn't improve the performance. Felt 2/3 epochs will get the desired output.


