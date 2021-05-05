#Import modules
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter = ',')
        next(reader)
        label = []
        temp_images = []
        for row in reader:
            label.append(row[0])
            image_data = row[1:785]
            image_data_array = np.array_split(image_data, 28)
            temp_images.append(image_data_array)
        images = np.array(temp_images).astype(float)
        labels = np.array(label).astype(float)
        return images, labels


training_images, training_labels = get_data('H:/Tensorflow_Coursera/signs_mnist/sign_mnist_train.csv')
testing_images, testing_labels = get_data('H:/Tensorflow_Coursera/signs_mnist/sign_mnist_test.csv')

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

#Expand dims
training_images = np.expand_dims(training_images, -1)
testing_images = np.expand_dims(testing_images, -1)
print(training_images.shape)
print(testing_images.shape)

#Data augmentation
train_datagen = ImageDataGenerator(rescale = (1.0/255),
                                   shear_range = 0.4,
                                   fill_mode = 'nearest',
                                   zoom_range = 0.4,
                                   rotation_range = 4,
                                   horizontal_flip= True,
                                   vertical_flip=True)
validation_datagen = ImageDataGenerator(rescale=(1.0/255))

#Creat model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'RMSprop', metrics = ['accuracy'])


history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=15,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)
model.save("my_model")

model.evaluate(testing_images, testing_labels)

#plot the chart for accuracy and loss on both training and validation

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()