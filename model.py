import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

correction = 0.1 # Correction +/- for left/right cameras

# Function which creates a list of image file-names,
# indicator for whether image is flipped or not (for data augmentation)
# and steering angle measurement
def load_filenames_and_measurements(folder):
	lines = []
	filenames_in_folder = []
	flipped_or_not = []
	measurements_in_folder = []

	# Parse driving log
	with open(folder + '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	for line in lines:
		# Center image
		filename = folder + '/' + line[0].strip()
		measurement = float(line[3])
		filenames_in_folder.append(filename)
		flipped_or_not.append(0)
		filenames_in_folder.append(filename)
		measurements_in_folder.append(measurement)
		# Flipped center image
		flipped_or_not.append(1)
		measurements_in_folder.append(measurement*-1.0)

		# Left image
		filename = folder + '/' + line[1].strip()
		filenames_in_folder.append(filename)
		flipped_or_not.append(0)
		filenames_in_folder.append(filename)
		measurements_in_folder.append(measurement +  correction)
		# Flipped left image
		flipped_or_not.append(1)		
		measurements_in_folder.append((measurement +  correction)*-1.0)		

		# Right image
		filename = folder + '/' + line[2].strip()
		filenames_in_folder.append(filename)
		flipped_or_not.append(0)
		filenames_in_folder.append(filename)
		measurements_in_folder.append(measurement -  correction)
		# Flipped right image		
		flipped_or_not.append(1)			
		measurements_in_folder.append((measurement -  correction)*-1.0)	

	return list(zip(filenames_in_folder, flipped_or_not, measurements_in_folder)) # Zipped list of image file name,
																				  #	flipped/not-flipped indicator and steering angle
# Generator function for parsing image files and
# storing them in numpy arrays for each batch
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images_batch = []
            angles_batch = []
            for batch_sample in batch_samples:
            	if (batch_sample[1] == 0):
            		images_batch.append(cv2.imread(batch_sample[0]))
            	else:
            		images_batch.append(cv2.flip(cv2.imread(batch_sample[0]),1))
            	angles_batch.append(batch_sample[2])

            # trim image to only see section with road
            X_train = np.array(images_batch)
            y_train = np.array(angles_batch)
            yield sklearn.utils.shuffle(X_train, y_train)

# List of folders which contains training data
# If the performance is poor on some section of the track
# we record more training data on that particular section
# and add the images folder from that run here.
folder_list = ['data_augment_2_challenge_track', 'data_augment_4_challenge_track', 'data_augment_5_challenge_track', 'data_augment_6_challenge_track', 'data_augment_7_challenge_track', 'data_augment_8_challenge_track', 'data_augment_9_challenge_track'] #['data', 'data_augment_1', 'data_augment_3'] #['data_augment_2_challenge_track', 'data_augment_4_challenge_track']

combined_data_set = []
# Loop through all the folders in the training set
# and aggregate the file-names and steering angle 
# measurements for center/left/right/swapped images
# within each folder
for folder in folder_list:
	combined_data_set_in_folder = load_filenames_and_measurements(folder)
	combined_data_set.extend(combined_data_set_in_folder)

# 80/20 split for training and validation
train_samples, validation_samples = train_test_split(combined_data_set, test_size=0.2)

# Use generator for training and validation sets to keep
# memory usage manageable
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Use Nvidia's autonomous driving network architecture
model = Sequential()

# Normalize and crop
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Conv layer 1
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

# Conv layer 2
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

# Conv layer 3
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

# Conv layer 4
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

# Conv layer 5
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())

# Fully connected 1 with dropout for regularization
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# Fully connected 2 with dropout for regularization
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# Fully connected 3 with dropout for regularization
model.add(Dense(10))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1))

# Use Adam optimizer with MSE loss metric
model.compile(loss='mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=12)

# Save model
model.save('model.h5')