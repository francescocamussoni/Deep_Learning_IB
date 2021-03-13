"""
date: 7-10-2020
File: ej7.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from textwrap import wrap
import seaborn as sns
sns.set(style='whitegrid')

#cargo datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=10000, stratify=y_train)

#normalizo y agrego ruido
x_train = x_train/255
x_val = x_val/255
x_test = x_test/255

x_train_r = x_train + np.random.normal(loc=0, scale=0.5, size=x_train.shape)
x_test_r = x_test + np.random.normal(loc=0, scale=0.5, size=x_test.shape)
x_val_r = x_val + np.random.normal(loc=0, scale=0.5, size=x_val.shape)

x_train_r = np.clip(x_train_r, 0, 1)
x_test_r = np.clip(x_test_r, 0, 1)
x_val_r = np.clip(x_val_r, 0, 1)

fig = plt.figure()
ax1 = plt.subplot(121)
ax1.imshow(x_train[0], cmap='binary')
ax1.set_xticks([])
ax1.set_yticks([])
plt.title('Imagen sin ruido', fontsize=15)
ax2 = plt.subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(x_train_r[0], cmap='binary')
plt.title('Imagen con ruido', fontsize=15)
plt.suptitle('Primer número del set de MNIST', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=1)
plt.savefig('ej_7_im_ruido.pdf', format='pdf')
plt.show()

#Parametros
lr=1e-3
rf=1e-3
batch_size=100
epochs=50

#Armo el autoencoeder (me super basé en https://blog.keras.io/building-autoencoders-in-keras.html)
input_img = keras.layers.Input(shape=x_train[0].shape)
x = keras.layers.Conv2D(32,
    (3, 3),
    activation='relu',
    padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2),
    padding='same')(x)
x = keras.layers.Conv2D(16,
    (3, 3),
    activation='relu',
    padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2),
    padding='same')(x)
x = keras.layers.Conv2D(1,
    (3, 3),
    activation='relu',
    padding='same')(x)
encoded = keras.layers.MaxPooling2D((2, 2),
    padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = keras.layers.Conv2D(1,
    (3, 3),
    activation='relu',
    padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(16,
    (3, 3),
    activation='relu',
    padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(32,
    (3, 3),
    activation='relu')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1,
    (3, 3),
    activation='sigmoid',
    padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss='binary_crossentropy',
    metrics=['mse'])

autoencoder.save_weights('weights_ej7')

##################################
    #Entrenamiento sin ruido
##################################

history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val))

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
ax2.scatter(np.arange(epochs), 1-np.array(history.history['mse']), marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), 1-np.array(history.history['mse']), color='green', label='Training acc')
ax2.scatter(np.arange(epochs), 1-np.array(history.history['val_mse']), marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), 1-np.array(history.history['val_mse']), color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(1-np.array(history.history['mse'])), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(1-np.array(history.history['val_mse'])), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de entrenamiento. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej7_sinruido.pdf', format='pdf')
plt.show()
plt.close()

decoded_imgs = autoencoder.predict(x_test)
encoder = keras.Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 5))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i==5:
        ax.set_title('Imagen original', fontsize=14, wrap=True)

    # Display with noise
    ax = plt.subplot(3, n, i + n)
    plt.imshow(encoded_imgs[i].reshape(4, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i==5:
        ax.set_title('Imagen decodificada', fontsize=14, wrap=True)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i==5:
        ax.set_title('Imagen reconstruida', fontsize=14, wrap=True)

plt.suptitle('Imagenes originales, codificadas y decodificadas para los datos de test.', fontsize=16)
plt.savefig('Ej7_encoded_decoded.pdf', format='pdf')
plt.show()

##################################
    #Entrenamiento con ruido
##################################

autoencoder.load_weights('weights_ej7')

history = autoencoder.fit(x_train_r, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val))

decoded_imgs = autoencoder.predict(x_test)

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
ax2.scatter(np.arange(epochs), 1-np.array(history.history['mse']), marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), 1-np.array(history.history['mse']), color='green', label='Training acc')
ax2.scatter(np.arange(epochs), 1-np.array(history.history['val_mse']), marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), 1-np.array(history.history['val_mse']), color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(1-np.array(history.history['mse'])), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(1-np.array(history.history['val_mse'])), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de entrenamiento. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej7_conruido.pdf', format='pdf')
plt.show()
plt.close()

n = 10
plt.figure(figsize=(20, 5))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i==5:
        ax.set_title('Imagen original', fontsize=14, wrap=True)

    # Display with noise
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test_r[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i==5:
        ax.set_title('Imagen con ruido', fontsize=14, wrap=True)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i==5:
        ax.set_title('Imagen reconstruida', fontsize=14, wrap=True)

plt.suptitle('Imagenes originales, con ruido y decodificadas sin ruido para los datos de test.', fontsize=16)
plt.savefig('Ej7_decoded.pdf', format='pdf')
plt.show()
