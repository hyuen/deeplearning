# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, MaxPooling2D
import numpy
import matplotlib.pyplot as plt



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()

model.add(Dense(4, input_dim=8, init='uniform', activation='relu'))
#model.add(MaxPooling1D(pool_length=3))
#model.add(Dense(4, init='uniform', activation='relu'))
#model.add(Dense(16, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(4, init='uniform', activation='relu'))
#model.add(Dense(16, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(16, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(4, init='uniform', activation='relu'))
#model.add(Dense(16, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(4, init='uniform', activation='relu'))
#model.add(Dense(16, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, nb_epoch=5000, batch_size=10,  verbose=1)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)
model.save('myfirst.h5')
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
