import h5py
import scipy.io as sio
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential
import numpy as np

def main():
  f = h5py.File('digitStruct.mat','r')
  #print f['digitStruct']['name'].keys()

  print "groups", f.keys()
  for k,v in f['digitStruct'].attrs.iteritems():
    print k, v

  g = f['digitStruct']['name']
  print "thisgroup", len(g), g.shape
  for i in g:
    #print dir(i)
    print i, i.data, i.dtype, i.var, i.item, i.size
    print dir(i)
    break


def main2():
  data = sio.loadmat('train_32x32.mat')
  #print data['X'].shape, len(data['X']), data['X'].shape[-1]
  #print data['y'].shape, len(data['y'])

  x_train = data['X'] / 255
  y_train = data['y']
  y_train = y_train.reshape(y_train.shape[0]) - 1
  x_train = x_train.astype('float32')
  img_rows = 32
  img_cols = 32
  print x_train.shape, y_train.shape
  
  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[3], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

  num_classes = 10
  y_train = keras.utils.to_categorical(y_train, num_classes)
  print x_train.shape
  print y_train.shape
  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
  #model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(16, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=100, epochs=10000, verbose = 1, validation_split=0.1)
  #            x_test, y_test)
  
main2()
