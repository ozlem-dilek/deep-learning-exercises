from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

from google.colab import drive
drive.mount('/content/drive')

model = Sequential()

model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(64, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(192, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.summary()

train_path = '/content/drive/MyDrive/muhendislik-tasarimi/dataset_yeni2/dataset/train'
test_path = '/content/drive/MyDrive/muhendislik-tasarimi/dataset_yeni2/dataset/test'
val_path = '/content/drive/MyDrive/muhendislik-tasarimi/dataset_yeni2/dataset/val'

train_data_gen = ImageDataGenerator(rescale=1/255) # Ölçeklendirme
test_data_gen = ImageDataGenerator(rescale=1/255)
val_data_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(224,224), # Görüntü boyutları
        batch_size=16,
        class_mode='categorical')  # Sınıf modu, çok sınıflı bir sınıflandırma olduğu için 'categorical'

test_generator = test_data_gen.flow_from_directory(
    test_path,
    target_size=(224,224),  # Görüntü boyutları
    batch_size=16,
    class_mode='categorical'  # Sınıf modu, çok sınıflı bir sınıflandırma olduğu için 'categorical'
)

val_generator = val_data_gen.flow_from_directory(
    val_path,
    target_size=(224,224),  # Görüntü boyutları
    batch_size=16,
    class_mode='categorical'  # Sınıf modu, çok sınıflı bir sınıflandırma olduğu için 'categorical'
)

checkpoint = ModelCheckpoint(
    f'/content/drive/MyDrive/muhendislik-tasarimi/AlexNet/AlexNet-model_4.h5',
     monitor='val_accuracy',
     verbose=1,
     save_best_only=True,
     mode = 'max'
)

earlystop = EarlyStopping(monitor='val_accuracy',
                          patience=5,
                          verbose=1,
                          mode = 'max')

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Adam, Adadelta'ya göre daha iyi bir accuracy ve loss verdi.

# Modeli eğitme
results = model.fit(train_generator, epochs=15, verbose=1, callbacks=[checkpoint,earlystop], validation_data= val_generator)

import matplotlib.pyplot as plt
train_loss = results.history['loss']
train_accuracy = results.history['accuracy']

# train loss ve train accuracy görselleştirilmesi
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, train_accuracy, 'r', label='Training accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()

plt.show()

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

#model ile tahmin (test verileriyle)
predictions = model.predict(test_generator)

num_samples_to_visualize = 10
test_labels = []
predicted_labels = []

for i, (_, labels) in enumerate(test_generator):
    test_labels.extend(labels.argmax(axis=1))  # Gerçek etiketler
    predicted_labels.extend(predictions.argmax(axis=1))  # Tahmin edilen etiketler
    if i == num_samples_to_visualize - 1:
        break

plt.figure(figsize=(12, 8))
for i in range(num_samples_to_visualize):
    plt.subplot(5, 2, i + 1)
    plt.imshow(test_generator[i][0][0])
    plt.title(f'Real: {test_labels[i]}, Predicted: {predicted_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

