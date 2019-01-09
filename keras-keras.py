import time
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras import optimizers

#with tf.device('/cpu:0'):
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential()
model.add(Flatten(data_format=None))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


start_time = time.time()
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
timer = time.time()
print(timer - start_time)