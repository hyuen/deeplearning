import h5py
import scipy.io as sio
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential


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
  data = sio.loadmat('../train_32x32.mat')
  print data['X'].shape, len(data['X']), data['X'].shape[-1]
  print data['y'].shape, len(data['y'])

  x_train = []
  y_train = []
  for i in range(data['X'].shape[-1]):
    img = data['X'][:,:,:,i]
    d = data['y'][i]
    x_train.append(img)
    y_train.append(d)


  model = Sequential()
  input_shape = (32,32,3)
  num_classes = 10
  
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=10)
  
main2()
