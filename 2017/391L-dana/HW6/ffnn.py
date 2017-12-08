from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import pickle

# Reference: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# Reference: https://www.learnopencv.com/image-classification-using-feedforward-neural-network-in-keras/
# (include a nice pic about Keras workflow, Dense layer explanation)

# Load the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# The data consists of handwritten numbers ranging from 0 to 9, along with their ground truth.
# It has 60,000 train samples and 10,000 test samples. Each sample is a 28x28 grayscale image.

print('Training data shape : ', train_images.shape, train_labels.shape)
print('Testing data shape : ', test_images.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Enable this knob if you want to see some sample training examples.
show_img = False
if show_img == True:
    plt.figure(figsize=[10, 5])

    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(train_images[0, :, :], cmap='gray')
    plt.title("Ground Truth : {}".format(train_labels[0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_images[0, :, :], cmap='gray')
    plt.title("Ground Truth : {}".format(test_labels[0]))
    plt.show()

# Change from matrix to array of dimension 28x28 to array of dimention 784
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

# Change the labels from integer to categorical data
train_labels_one_hot = keras.utils.to_categorical(train_labels)
test_labels_one_hot = keras.utils.to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

# This knob is used because we cannot see the plot from the remote server.
train = True
if train == True:
    # Define the FFNN with two layers
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dimData,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    print('Train...')
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-val.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    callbacks_list = [checkpoint]
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                       validation_data=(test_data, test_labels_one_hot), callbacks=callbacks_list)

    [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

    print(history.history.keys())

    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Predict the most likely class
    model.predict_classes(test_data[[0], :])

    # Predict the probabilities for each class
    model.predict(test_data[[0], :])
else:
    history_history = pickle.load(open('trainHistoryDict', 'rb'))
    # Plot the Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history_history['loss'], 'r', linewidth=3.0)
    plt.plot(history_history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # Plot the Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history_history['acc'], 'r', linewidth=3.0)
    plt.plot(history_history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()

