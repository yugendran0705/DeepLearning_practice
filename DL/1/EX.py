import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#plt.matshow(x_train[0])
#plt.show()
#print(y_train[0])

x_train = x_train/255
x_test = x_test/255
'''x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
print(x_train_flattened.shape)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flattened, y_train, epochs=5)
model.evaluate(x_test_flattened, y_test)
y_predicted = model.predict(x_test_flattened)
print(y_predicted[0])
print(np.argmax(y_predicted[0]))
plt.matshow(x_test[0])
plt.show()'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)