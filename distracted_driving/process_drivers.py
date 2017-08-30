#!/usr/bin/env python
import numpy as np
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import csv
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
    print "a", x.shape
    x = x - vgg_mean
    print x.shape
    return x [:, ::-1] # reverse axis rgb->bgr
    #return x


model = Sequential()
model.add(Lambda(vgg_preprocess, input_shape=(32,32,3), output_shape=(32,32,3)))
model.add(Conv2D(16, 3, activation='relu'))

#model.add(Conv2D(16, 3, input_shape=(32, 32, 3), activation='relu'))
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
model.load_weights("weights-improvement-00-0.59.hdf5")
print model.summary()

train_datagen = ImageDataGenerator(rescale = 1,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                               )

test_datagen = ImageDataGenerator(rescale = 1)

training_set = train_datagen.flow_from_directory('sub/train',
                                                 target_size = (32, 32),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('sub/val',
                                            target_size = (32, 32),
                                            batch_size = 16,
                                            class_mode = 'categorical')

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['categorical_accuracy'])

filepath="weights-improvement-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(training_set,
                         steps_per_epoch = 20000/16,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000/16, callbacks=callbacks_list)
print model.summary()

b=16
submit_datagen = ImageDataGenerator(rescale= 1) #'sub/test'
submit_set = submit_datagen.flow_from_directory('sub/test', target_size= (32,32), batch_size=b,
                                                class_mode=None, shuffle=False)

print submit_set.samples, submit_set.batch_size, submit_set.samples / b + 1
preds = model.predict_generator(submit_set, 
                                submit_set.samples / b + 1, verbose=1
                               )
print len(preds)

with open('submission.csv', 'w') as csvfile:
  csvw = csv.writer(csvfile)
  csvw.writerow(['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
  for fn, sc in zip(submit_set.filenames, preds):
    csvw.writerow([fn.split('/')[1]] + list(sc))




