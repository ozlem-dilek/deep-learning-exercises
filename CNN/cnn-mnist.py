from keras.datasets import mnist
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#x = images, y = labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype("float32")/255 #normalization

x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype("float32")/255 #normalization

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=128, epochs=2)

_, accuracy = model.evaluate(x_test, y_test, batch_size = 64)

print("accuracy = ", accuracy)