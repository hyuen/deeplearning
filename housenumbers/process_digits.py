import h5py
import scipy.io as sio
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras import backend as K
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras.optimizers
print dir(keras.optimizers)
import sys
import numpy as np

def addconvblock(model, layers, filters):
  for i in range(layers):
    #model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(filters, (3,3), activation='relu'))
  model.add(MaxPooling2D((2,2))) # , strides=(2,2)))
            

def main2():
  data = sio.loadmat('train_32x32.mat')

  x_train = data['X']
  y_train = data['y']
  print x_train.shape, y_train.shape

  y_train = y_train.reshape(y_train.shape[0]) - 1
  #x_train = x_train.astype('float32') / 255
  img_rows = 32
  img_cols = 32
  print x_train.shape, y_train.shape
  
  #  if K.image_data_format() == 'channels_first':
  #  x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
  #  input_shape = (3, img_rows, img_cols)
  #else:
  #x_train = x_train.reshape(x_train.shape[3], img_rows, img_cols, 3)
  input_shape = (img_rows, img_cols, 3)
  x2 = np.array([x_train[:,:,:,i] for i in range(x_train.shape[-1])])
  x_train = x2

  num_classes = 10
  y_train = keras.utils.to_categorical(y_train, num_classes)
  print x_train.shape
  print y_train.shape
  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

  addconvblock(model, 2, 32)
  model.add(Dropout(0.25))
  addconvblock(model, 5, 64)
  model.add(Dropout(0.25))

  model.add(Flatten())

  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.25))

  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  #model.load_weights("weights-improvement-199-0.95.hdf5")
  
  opt = keras.optimizers.Adadelta(lr=0.001)
  model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  print model.summary()
  #sys.exit(0)
  plot_model(model, to_file='model.png', show_shapes=True)
  filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  # sys.exit(0)
  model.fit(x_train, y_train,
            batch_size=512, epochs=200, verbose=1, validation_split=0.1,
            callbacks=callbacks_list, shuffle=True)
  
main2()
