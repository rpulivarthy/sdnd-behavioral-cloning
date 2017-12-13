import csv
import numpy as np
import cv2
# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Lambda
# from keras.layers.convolutional import Convolution2D
# from keras.layers.pooling import MaxPooling2D


def get_image(file_path):
    filename = file_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    return cv2.imread(current_path)

def load_data(left_correction, right_correction):
    """
    Load the data into numpy array. Load all the images; left center and right for better training.
    """
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

    images =[]
    measurements = []

    for line in lines:        
        measurement = float(line[3])
        for i in range(3):
            images.append(get_image(line[i]))
            if i == 1:
                measurement = measurement + left_correction
            elif i == 2:
                measurement = measurement + right_correction
            measurements.append(measurement)
    
    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train

 
def generator(samples, batch_size=32):
    
# def build_model():
#     """
#         NVIDIA model
#     """
#     model = Sequential()
#     model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
#     model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
#     model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
#     model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
#     model.add(Conv2D(64, 3, 3, activation='elu'))
#     model.add(Conv2D(64, 3, 3, activation='elu'))
#     model.add(Dropout(args.keep_prob))
#     model.add(Flatten())
#     model.add(Dense(100, activation='elu'))
#     model.add(Dense(50, activation='elu'))
#     model.add(Dense(10, activation='elu'))
#     model.add(Dense(1))
#     model.summary()
#     return model

# def train_model():
    
#     model.compile(loss='mse', optimizer='adam')
#     model.compile(X_train, y_train, validation_spilit = 0.2, shuffle= True, nb_epoch= 5)

#     model.save('model.h5')


load_data(0.5,-0.5)
