from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import datetime
import os

# Loading the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initializing the hyper-parameter
epochs = 10
batch_size = 128
# keep_probability = 0.3
learning_rate = 0.001
weight_decay = 1e-4

# Training the model
# Layer 1 convolution, max-pooling, batch normalization and drop-out
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))

# Layer 2 convolution, max-pooling, batch normalization and drop-out
model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.3))

# Layer 3 convolution, max-pooling, batch normalization and drop-out
model.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.4))

# Layer 4 convolution, max-pooling, batch normalization and drop-out
model.add(keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.5))

# Flattening the layer
model.add(keras.layers.Flatten())

# Output layer
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# Initializing the RMSprop optimizer
rms = keras.optimizers.RMSprop(lr=learning_rate, decay=1e-6)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

# Creating a log file
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training the model
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback],
          validation_data=(test_images, test_labels))

# Importing the model to json file for future use
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

test_loss, test_acc, topk_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTest k accuracy:', topk_acc)
