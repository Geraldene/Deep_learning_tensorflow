#Import modules
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

#Load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#Normalize data
training_images = training_images /255.0
test_images = test_images /255.0

#callbacks
class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.98):
            print('Reached an accuracy of 98%')
            self.model.stop_training = True
callbacks = mycallbacks()


#Create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(loss= 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history = model.fit(training_images, training_labels, epochs = 100, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

#Test
classifications = model.predict(test_images)
print(classifications[0])

print(test_labels[0])