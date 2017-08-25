#!/usr/bin/env python
import numpy as np
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import csv

model = Sequential()

model.add(Conv2D(16, 3, input_shape=(32, 32, 3), activation='relu'))

model.add(Conv2D(16, 3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(32, 3,  activation = 'relu'))       
model.add(Conv2D(32, 3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) 

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#weights-improvement-55-0.99.hdf5
model.load_weights("weights-improvement-15-0.95.hdf5")
print model.summary()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                               )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (32, 32),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('val',
                                            target_size = (32, 32),
                                            batch_size = 16,
                                            class_mode = 'categorical')

model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
              loss='categorical_crossentropy', metrics=['categorical_accuracy'])

filepath="weights-improvement-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(training_set,
                         steps_per_epoch = 14000/16,
                         epochs = 0,
                         validation_data = test_set,
                         validation_steps = 7000/16, callbacks=callbacks_list)
print model.summary()

b=128
submit_datagen = ImageDataGenerator(rescale= 1./255) #'sub/test'
submit_set = submit_datagen.flow_from_directory('sub/test', target_size= (32,32), batch_size=b, class_mode=None, shuffle=False)

print submit_set.samples, submit_set.batch_size, submit_set.samples / b + 1
preds = model.predict_generator(submit_set, 
                                  submit_set.samples / b + 1, workers=4
                                  )
print len(preds)

with open('submission.csv', 'w') as csvfile:
  csvw = csv.writer(csvfile)
  csvw.writerow(['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
  for fn, sc in zip(submit_set.filenames, preds):
    csvw.writerow([fn.split('/')[1]] + list(sc))




