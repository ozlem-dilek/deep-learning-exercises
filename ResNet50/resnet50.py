from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_path = 'dataset/train'
test_path = 'dataset/test'
val_path = 'dataset/val'

train_data_gen = ImageDataGenerator(rescale=1/255)
test_data_gen = ImageDataGenerator(rescale=1/255)
val_data_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(224,224),
        batch_size=16,
        class_mode='categorical')  

test_generator = test_data_gen.flow_from_directory(
    test_path,
    target_size=(224,224), 
    batch_size=16,
    class_mode='categorical'
)

val_generator = val_data_gen.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical' 
)

checkpoint = ModelCheckpoint(
    f'resnet50_model.h5',
     monitor='val_accuracy',
     verbose=1,
     save_best_only=True,
     mode = 'max'
)

earlystop = EarlyStopping(monitor='val_accuracy',
                          patience=5,
                          verbose=1,
                          mode = 'max')

model = Sequential()
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.add(resnet)

model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
num_classes = train_generator.num_classes
model.add(Dense(num_classes, activation='softmax'))

# temel katman dondurma
for layer in resnet.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

result = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=25
)

import matplotlib.pyplot as plt
train_loss = result.history['loss']
train_accuracy = result.history['accuracy']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, train_accuracy, 'r', label='Training accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()

plt.show()

model.save("resnet50_model.keras")

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = model.predict(test_generator)

num_samples_to_visualize = 10
test_labels = []  
predicted_labels = [] 

# Test verilerinden örnekleri al
for i, (_, labels) in enumerate(test_generator):
    test_labels.extend(labels.argmax(axis=1))  
    predicted_labels.extend(predictions.argmax(axis=1)) 
    if i == num_samples_to_visualize - 1:
        break

# Sonuçları görselleştir
plt.figure(figsize=(12, 8))
for i in range(num_samples_to_visualize):
    plt.subplot(5, 2, i + 1)
    plt.imshow(test_generator[i][0][0])
    plt.title(f'Real: {test_labels[i]}, Predicted: {predicted_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

