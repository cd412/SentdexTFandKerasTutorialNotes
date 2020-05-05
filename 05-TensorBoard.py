import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import os
import time
os.chdir(r'C:\Users\student\Documents\repos\_Projects\SentdexTFandKeras')

# Import data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
y = np.array(y)

# Normalize / Scale Data
X = X/255.0

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir='logs\\{}'.format(NAME))
            print(NAME)
            
            # Model Building
            model = Sequential()

            # Conv Layers
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:])) #3x3 window
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3))) #3x3 window
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            # Dense Layers
            for l in range(dense_layer):
                # Final layer
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # Output layer
            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])
