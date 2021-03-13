"""
date: 4-11-2020
File: ej3.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
from os import listdir
import numpy as np
from numpy import asarray
from numpy import save
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, Xception, MobileNet
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

#esto fue para tener todo en 128 128
folder = 'train/'
photos, labels = list(), list()
for file in listdir(folder):
    output = 0.0
    if file.startswith('cat'):
        output = 1.0
    photo = load_img(folder + file, target_size=(128, 128))
    photo = img_to_array(photo)
    photos.append(photo)
    labels.append(output)
photos = asarray(photos).astype(np.uint8)
labels = asarray(labels).astype(np.uint8)
print(photos.shape, labels.shape)
save('cd_128_photos.npy', photos)
save('cd_128_labels.npy', labels)

########################################
                #######
                       #
                 #######
                #      #
                 ########
########################################
x=np.load('cd_128_photos.npy').astype(np.float32)
y=np.load('cd_128_labels.npy').astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2500, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=2500, stratify=y_train)

img_mean = x_train.mean(axis=0)
img_std = x_train.std(axis=0)
img_std[img_std==0]=1

x_train -= img_mean
x_train /= img_std
x_test -= img_mean
x_test /= img_std
x_val -= img_mean
x_val /= img_std

idg = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0,
    zoom_range=0,
    fill_mode='reflect',
    horizontal_flip=True
)

lr=1e-4
rf=0
epochs=50
batch_size=100
df=0.2
n_entries=2500

#Me baso en el libro de Chollet Y https://keras.io/guides/transfer_learning/ (que tambien escribio Chollet)
conv_base = MobileNet(weights="imagenet", input_shape=x_train[0].shape, include_top=False)
conv_base.trainable = False
inputs = keras.Input(shape=x_train[0].shape)
x = conv_base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(df)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name='acc')],
)

history = model.fit(idg.flow(x_train[:n_entries], y_train[:n_entries], batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=n_entries / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('El accuracy sobre los datos de validacion fue= ', acc)

#fine tunning
conv_base.trainable = True
model.summary()
epochs_ft=10

model.compile(
    optimizer=keras.optimizers.Adam(lr/10),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name='acc')],
)

history_ft = model.fit(idg.flow(x_train[:n_entries], y_train[:n_entries], batch_size=batch_size),
    epochs=epochs_ft,
    steps_per_epoch=n_entries / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('El accuracy sobre los datos de validacion fue= ', acc)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs+epochs_ft), np.append(history.history['loss'],history_ft.history['loss']), color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs+epochs_ft), np.append(history.history['loss'],history_ft.history['loss']), color='red', label='Training loss')
ax1.scatter(np.arange(epochs+epochs_ft), np.append(history.history['val_loss'],history_ft.history['val_loss']), color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs+epochs_ft), np.append(history.history['val_loss'],history_ft.history['val_loss']), color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(np.append(history.history['loss'],history_ft.history['loss'])), 0, epochs+epochs_ft, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(np.append(history.history['val_loss'],history_ft.history['val_loss'])), 0, epochs+epochs_ft, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs+epochs_ft), np.append(history.history['acc'], history_ft.history['acc']), marker='o', alpha=0.5)
ax2.plot(np.arange(epochs+epochs_ft), np.append(history.history['acc'], history_ft.history['acc']), color='green', label='Training acc')
ax2.scatter(np.arange(epochs+epochs_ft), np.append(history.history['val_acc'], history_ft.history['val_acc']), marker='.', alpha=0.5)
ax2.plot(np.arange(epochs+epochs_ft), np.append(history.history['val_acc'], history_ft.history['val_acc']), color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(np.append(history.history['acc'], history_ft.history['acc'])), 0, epochs+epochs_ft, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(np.append(history.history['val_acc'], history_ft.history['val_acc'])), 0, epochs+epochs_ft, color='orange', linestyles='dashdot', alpha=0.6)
ax2.vlines(epochs, min(np.append(history.history['acc'], history_ft.history['acc'])),max(np.append(history.history['val_acc'], history_ft.history['val_acc'])), linestyles='dashed')
plt.title('Resultados de entrenamiento transfer learning+finetunning. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej3_a_finetunning.pdf', format='pdf')
plt.show()
plt.close()

##################################################################
                        #entreno de cero
##################################################################
lr=5e-3
rf=0
epochs=50
batch_size=100
df=0.2
n_entries=5000

model_sin_pretraining = MobileNet(weights=None, input_shape=x_train[0].shape, include_top=False)
inputs = keras.Input(shape=x_train[0].shape)
x = model_sin_pretraining(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(df)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model_sin_pretraining = keras.Model(inputs, outputs)

model_sin_pretraining.compile(
    optimizer=keras.optimizers.Adam(lr),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name='acc')],
)

history = model_sin_pretraining.fit(idg.flow(x_train[:n_entries], y_train[:n_entries], batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=n_entries / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='green', label='Training acc')
ax2.scatter(np.arange(epochs), history.history['val_acc'], marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de sin pretraining. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej3_a_catdogs_2.pdf', format='pdf')
plt.show()
plt.close()
########################################
                #
                #
                #######
                #      #
                #######
########################################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from textwrap import wrap
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sns.set(style='whitegrid')

# ENTRENO PRIMERO MI RED PARA EMINST
#cargo datos
train=np.loadtxt('emnist-letters-train.csv', delimiter = ',')
test=np.loadtxt('emnist-letters-test.csv', delimiter = ',')
n_clases=26
n_entries=60000 #para emular mnist
n_entries_test=15000
idx=np.random.randint(0, test.shape[0]-1, n_entries_test)#esto es porque estan ordenados...
x_train=train[:n_entries, 1:].reshape(n_entries, 28, 28, 1)
y_train=train[:n_entries, 0]-1
x_test=test[idx, 1:].reshape(n_entries_test, 28, 28, 1)
y_test=test[idx, 0]-1
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000, stratify=y_train)
x_train=x_train/255.
x_test=x_test/255.
x_val=x_val/255.

input_dim=x_train.shape[1]
lr=1e-4
rf=1e-2
batch_size=100
epochs=50

model_conv = keras.models.Sequential(name='Ejercicio_8_conv')
model_conv.add(keras.layers.Input(shape=x_train[0].shape))
model_conv.add(keras.layers.Conv2D(10,
    (24, 24),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same'))
model_conv.add(keras.layers.MaxPooling2D((2, 2),
    padding='same'))
model_conv.add(keras.layers.Conv2D(10,
    (12, 12),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same'))
model_conv.add(keras.layers.Flatten())
model_conv.add(keras.layers.Dense(n_clases,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf)))

model_conv.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
model_conv.summary()

history = model_conv.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, y_val),
    verbose=2)

model_conv.save_weights('weights_ej3_b')
model_conv.save('model_mnist.h5')

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='green', label='Training acc')
ax2.scatter(np.arange(epochs), history.history['val_acc'], marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de entrenamiento para EMNIST. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej3_b_emnist.pdf', format='pdf')
plt.show()
plt.close()

################################################
        #TRANSFIERO LO APRENDIDO A MNIST
################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=30000, stratify=y_train)

x_train = x_train/255
x_val = x_val/255
x_test = x_test/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_val = np.reshape(x_val, (len(x_val), 28, 28, 1))

idg = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0,
    zoom_range=0,
    fill_mode='reflect',
    horizontal_flip=True
)

lr=5e-4
rf=1e-4
epochs=50
batch_size=100
n_entries=1000

model_base = keras.Model(inputs=model_conv.inputs, outputs=model_conv.layers[-2].output)
model_base.trainable = False

inputs = keras.layers.Input(shape=x_train[0].shape)
x = model_base (inputs)
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.CategoricalCrossentropy(name='loss', from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')],
)

history = model.fit(idg.flow(x_train[:n_entries], y_train[:n_entries], batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(x_train[:n_entries]) / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)


###################################
          #fine tunning
###################################
model_base.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(lr),  # Low learning rate
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')],
)

epochs_ft=50

history_ft = model.fit(idg.flow(x_train[:n_entries], y_train[:n_entries], batch_size=batch_size),
    epochs=epochs_ft,
    steps_per_epoch=len(x_train[:n_entries]) / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)


fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs+epochs_ft), np.append(history.history['loss'],history_ft.history['loss']), color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs+epochs_ft), np.append(history.history['loss'],history_ft.history['loss']), color='red', label='Training loss')
ax1.scatter(np.arange(epochs+epochs_ft), np.append(history.history['val_loss'],history_ft.history['val_loss']), color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs+epochs_ft), np.append(history.history['val_loss'],history_ft.history['val_loss']), color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(np.append(history.history['loss'],history_ft.history['loss'])), 0, epochs+epochs_ft, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(np.append(history.history['val_loss'],history_ft.history['val_loss'])), 0, epochs+epochs_ft, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs+epochs_ft), np.append(history.history['acc'], history_ft.history['acc']), marker='o', alpha=0.5)
ax2.plot(np.arange(epochs+epochs_ft), np.append(history.history['acc'], history_ft.history['acc']), color='green', label='Training acc')
ax2.scatter(np.arange(epochs+epochs_ft), np.append(history.history['val_acc'], history_ft.history['val_acc']), marker='.', alpha=0.5)
ax2.plot(np.arange(epochs+epochs_ft), np.append(history.history['val_acc'], history_ft.history['val_acc']), color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(np.append(history.history['acc'], history_ft.history['acc'])), 0, epochs+epochs_ft, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(np.append(history.history['val_acc'], history_ft.history['val_acc'])), 0, epochs+epochs_ft, color='orange', linestyles='dashdot', alpha=0.6)
ax2.vlines(epochs, min(np.append(history.history['acc'], history_ft.history['acc'])),max(np.append(history.history['val_acc'], history_ft.history['val_acc'])), linestyles='dashed')
plt.title('Resultados de entrenamiento transfer learning+finetunning. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej3_b_finetunning.pdf', format='pdf')
plt.show()
plt.close()

##############################################
        #y si entreno de una con mnist?
##############################################
lr=1e-3
rf=1e-2
batch_size=100
epochs=75
n_entries=1000

model_sin_pretraining = keras.models.Sequential(name='ej3_mnist')
model_sin_pretraining.add(keras.layers.Input(shape=x_train[0].shape))
model_sin_pretraining.add(keras.layers.Conv2D(10,
    (24, 24),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same'))
model_sin_pretraining.add(keras.layers.MaxPooling2D((2, 2),
    padding='same'))
model_sin_pretraining.add(keras.layers.Conv2D(10,
    (12, 12),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same'))
model_sin_pretraining.add(keras.layers.Flatten())
model_sin_pretraining.add(keras.layers.Dense(10,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf)))

model_sin_pretraining.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
model_sin_pretraining.summary()

history = model_sin_pretraining.fit(idg.flow(x_train[:n_entries], y_train[:n_entries], batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(x_train[:n_entries]) / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='green', label='Training acc')
ax2.scatter(np.arange(epochs), history.history['val_acc'], marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de sin pretraining. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej3_b_mnist.pdf', format='pdf')
plt.show()
plt.close()
