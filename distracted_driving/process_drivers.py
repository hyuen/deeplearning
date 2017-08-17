#!/usr/bin/env python

from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


model = Sequential()

model.add(Conv2D(16, 3, input_shape=(32, 32, 3), activation='relu'))

model.add(Conv2D(16, 3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(32, 3,  activation = 'relu'))       
model.add(Conv2D(32, 3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) 

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.load_weights("weights-improvement-06-0.97.hdf5")
print model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                               )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (32, 32),
                                                 batch_size = 4,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('val',
                                            target_size = (32, 32),
                                            batch_size = 16,
                                            class_mode = 'categorical')

model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
              loss='categorical_crossentropy', metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(training_set,
                         steps_per_epoch = 14000/16,
                         epochs = 200,
                         validation_data = test_set,
                         validation_steps = 7000/16, callbacks=callbacks_list)


