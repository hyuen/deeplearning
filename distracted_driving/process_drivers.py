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

import csv

model = Sequential()

model.add(Conv2D(16, 3, input_shape=(32, 32, 3), activation='relu'))

model.add(Conv2D(16, 3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(32, 3,  activation = 'relu'))       
model.add(Conv2D(32, 3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) 

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#weights-improvement-55-0.99.hdf5
#model.load_weights("weights-improvement-55-0.99.hdf5")
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
              loss='categorical_crossentropy', metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(training_set,
                         steps_per_epoch = 14000/16,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 7000/16, callbacks=callbacks_list)
print model.summary()
sys.exit(1)


submit_datagen = ImageDataGenerator(rescale= 1./255) #'sub/test'
submit_set = submit_datagen.flow_from_directory('sub/test', target_size= (32,32), batch_size=1, class_mode=None)

with open('submission.csv', 'w') as csvfile:
  csvw = csv.writer(csvfile)
  ct = 0
  for i in submit_set:
    idx = (submit_set.batch_index -1 ) * submit_set.batch_size
    ct = ct + 1
    if ct > 79726:
      break
    #if idx < 0:
    #  break
    if idx % 100 == 0:
      print idx
    fnames = submit_set.filenames[idx: idx+ submit_set.batch_size]
    res = np.round(model.predict(i), decimals=0)
    #print fnames, res
    #break

    for fn, sc in zip(fnames,res):
      #print fn, sc
      csvw.writerow([fn.split('/')[1]] + sc.tolist())


