from keras.models import Sequential
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.applications import VGG16
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np


#plant_village datasetinin yüklenmesi ve train/test olarak ayrılması (%80 train, %20 test)
data, info = tfds.load('plant_village', split='train', with_info=True)

train_samples = int(0.8 * 54303)
train_data = data.take(train_samples)
test_data = data.skip(train_samples)

#train ve test datalarını yeniden boyutlandırıp numpy dizisi haline dönüştürülmesi

resized_train_img = []
for i in train_data:
  img = i['image']
  resized_train = cv2.resize(img.numpy(), (128, 128))
  resized_train_img.append(resized_train)

resized_train_images_np = np.array(resized_train_img)

train_labels = []
for example in train_data:
    label = example['label']
    train_labels.append(label)

train_labels_np = np.array(train_labels)

resized_test_img = []
for i in test_data:
  img = i['image']
  resized_test = cv2.resize(img.numpy(), (128, 128))
  resized_test_img.append(resized_test)

resized_test_images_np = np.array(resized_test_img)

#vgg-16 modelinin tanımlanması
vgg = VGG16(
    input_shape=(128,128,3),
    weights='imagenet',
    include_top=False)

classesNum = 38

#yeni modelin oluşturulup katmanların eklenmesi
model = Sequential()
model.add(vgg)
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) #Katmanlar arası delay
model.add(Dense(classesNum , activation='softmax'))
print (model.summary())

#model derleme
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model kontrolü için modelcheckpoint ve earlystopping
checkpoint = ModelCheckpoint(
    f'/content/drive/MyDrive/deep-learning-exercies/VGG/{datetime.datetime.now()}.h5',
     monitor='val_accuracy',
     verbose=1,
     save_best_only=True
)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5 , verbose=1)

#one-hot-encoding
train_labels_np = to_categorical(train_labels_np, num_classes=38)

#modelin eğitilmesi
result = model.fit(resized_train_images_np, train_labels_np, batch_size=64, epochs=15, verbose=1 , callbacks=[checkpoint,earlystop])