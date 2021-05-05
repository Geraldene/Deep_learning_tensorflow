import os
from typing import Optional, Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(weights='imagenet', input_shape= (150, 150, 3), include_top = False )

for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()

  
last_layer = pre_trained_model.get_layer('mixed7')
last_output= last_layer.output



from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.optimizers import RMSprop

#  Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# # Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# # Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# # Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)


model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics = ['accuracy'])

train_dir = 'H:/Tensorflow_Coursera/horse_humans/horse-or-human/train/'
validation_dir = 'H:/Tensorflow_Coursera/horse_humans/horse-or-human/validation/'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

#callbacks

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            print('Reached 99% accuracy so cancelling training')
            self.model.stop_training= True

callbacks = myCallback()

#Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = (1.0/255),
                                   shear_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   zoom_range =0.4,
                                   fill_mode ='nearest',
                                   rotation_range = 2)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = (1.0/255))

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150, 150),
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150, 150),
                                                        class_mode = 'binary')

history = model.fit (train_generator, epochs = 100, validation_data = validation_generator, callbacks=[callbacks])

#plotting
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='training_acc')
plt.plot(epochs, val_acc, 'b', label = 'validation_acc')
plt.title('Training vs Validation accuracy')
plt.show()

plt.plot(epochs, loss, 'r', label='training_loss')
plt.plot(epochs, val_loss, 'b', label = 'validation_loss')
plt.title('Training vs Validation loss')
plt.show()


