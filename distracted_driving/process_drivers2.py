#!/usr/bin/env python
import numpy as np
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model

import csv

model = VGG16(weights='imagenet')

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                               )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('val',
                                            target_size = (224, 224), 
                                            batch_size = 16,
                                            class_mode = 'categorical')

#last = model.output
#x = Flatten()(last)
#x = Dense(10, activation='softmax')(x)
print model.summary()
model = Model(model.input, x)

model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(training_set,
                         steps_per_epoch = 14000/16,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 7000/16)
