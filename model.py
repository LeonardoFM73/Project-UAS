# Importing Keras for Image Classification
import os
from keras.layers import Dense,Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
import test

import warnings
warnings.filterwarnings("ignore")

# Fitting the Model
cnn = test.model.fit(test.train_data, 
                  steps_per_epoch = 2, 
                  epochs = 50, 
                  validation_data = test.val_data, 
                  validation_steps = 1,
                  callbacks = test.call_back )

model = load_model("./oral_cancer_best_model.hdf5")

# Checking the Accuracy of the Model 
accuracy = model.evaluate_generator(generator= test.test_data)[1] 
print(f"The accuracy of the model is = {accuracy*100} %")