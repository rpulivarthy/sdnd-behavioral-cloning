import csv
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


def get_image_path(file_path):
    """
    Get the image paths from the filepath in the csv
    """
    filename = file_path.split('/')[-1]
    return './data/IMG/' + filename

def load_data(correction):
    """
    Load the data into numpy array. Load all the images; left center and right for better training.
    Added a small correction for the left and right images.
    """
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

    image_paths =[]
    measurements = []

    for line in lines:        
        measurement = float(line[3])
        for i in range(3):
            image_paths.append(get_image_path(line[i]))
            if i == 1:
                measurement = measurement + correction
            elif i == 2:
                measurement = measurement - correction
            measurements.append(measurement)
    
    X_train = np.array(image_paths)
    y_train = np.array(measurements)
    return X_train, y_train

def nVidiaModel():
    """
    Creates a nVidia model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def preprocess_image(image):
    """
    This is used in the nVidia model
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def random_flip(image, measurement):
    """
    Flip the images and negate the measurements randomly
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        measurement = -measurement
    return image, measurement
 
def generator(samples, is_training, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                image = cv2.imread(imagePath)
                image = preprocess_image(image)
                if is_training:       
                    image, measurement = random_flip(image, measurement)
                images.append(image)
                angles.append(measurement)
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def main():
    image_paths, measurents = load_data(0.2)

    # split samples for training and validation
    samples = list(zip(image_paths, measurents))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples, True, 32)
    validation_generator = generator(validation_samples, False, 32)

    # Model Creation
    model = nVidiaModel()

    # checkpooint to save only the best from multiple epochs.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    # Compiling and training the model
    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(train_generator, 
                    samples_per_epoch=len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=6,
                    callbacks=[checkpoint], 
                    verbose=1)
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
