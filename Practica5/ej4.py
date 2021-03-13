"""
date: 4-11-2020
File: ej4.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras

########################################
                #######
                       #
                 #######
                #      #
                 ########
########################################

def compute_loss(input_image, index, layer):
    activation = feature_extractor(input_image)
    if layer=='filter':
        filter_activation = activation[:, 2:-2, 2:-2, index]
    elif layer=='class':
        filter_activation = activation[:, index]
        print(filter_activation)
    else:
        print('invalid class')
        return
    print(tf.reduce_mean(filter_activation))
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, index, learning_rate, layer, max_or_min):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, index, layer)

    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    if max_or_min=='max':
        img += learning_rate * grads
    elif max_or_min=='min':
        img -= learning_rate * grads
    else:
        print('invalid selection of learning')
        return
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(index, it=30, lr=10, layer='filter', max_or_min='max', type_img='noise', filename=None):
    # We run gradient ascent for 20 steps
    iterations = it
    learning_rate = lr
    if type_img=='noise':
        img = initialize_image()
    elif type_img=='specific':
        img = plt.imread(filename).astype(np.float32)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    else:
        print('invalida image type')
        return
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, index, learning_rate, layer, max_or_min)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img



img_width = 400
img_height = 400
layer_name = "conv5_block3_out"

model = keras.applications.ResNet50V2(weights="imagenet", include_top=True)
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

#### ploteo de los 64 filtros
all_imgs = []
for filter_index in range(64):
    print("Processing filter %d" % (filter_index,))
    loss, img = visualize_filter(filter_index, 30, 10, 'filter')
    all_imgs.append(img)

margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        img = all_imgs[i * n + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = img
keras.preprocessing.image.save_img("ej4_64filtros_conv5.pdf", stitched_filters)
### termina el ploteo de los 64 filtros

### ploteo al patón bauza
loss, img = visualize_filter(0, 1000, 10, 'filter', 'max', 'specific', 'paton_bauza.jpeg')
plt.imshow(img)
plt.title('El super hombre del cual se refería Nietszche a.k.a Patón Bauza luego de maximizar la activación', fontsize=14, wrap=True)
plt.axis('off')
plt.savefig('ej4_paton_conv5.pdf', format='pdf')
plt.show()

### ploteo un filtro especifico maximizando la activacion
loss, img = visualize_filter(0, 30, 10, 'filter', 'max')
plt.imshow(img)
plt.title('Filtro máxima para ruido como entrada', fontsize=14, wrap=True)
plt.axis('off')
plt.savefig('ej4_filter_max_conv5.pdf', format='pdf')
plt.show()

### ploteo un filtro especifico minimizando la activacion
loss, img = visualize_filter(0, 30, 10, 'filter', 'min')
plt.imshow(img)
plt.title('Filtro con activación mínima para ruido como entrada', fontsize=14, wrap=True)
plt.axis('off')
plt.savefig('ej4_filter_min_conv5.pdf', format='pdf')
plt.show()

### ploteo como se ve una clase para la red neuronal
img_width = 1080
img_height = 1080
layer_name = "probs"

model = keras.applications.ResNet50V2(weights="imagenet", include_top=True)
layer = model.get_layer(name=layer_name)

feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
loss, img = visualize_filter(283, 30, 100, 'class', 'max')
plt.imshow(img)
plt.title('Clase 283 (gato siames) con activación máxima para ruido como entrada', fontsize=14, wrap=True)
plt.axis('off')
plt.savefig('class.pdf', format='pdf')
plt.show()

########################################
                #
                #
                #######
                #      #
                #######
########################################
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from textwrap import wrap
import seaborn as sns
from tensorflow.keras.utils import to_categorical
sns.set(style='whitegrid')

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


''' Entreno la red 1 vez, desp la cargo
lr=5e-4
rf=1e-2
batch_size=100
epochs=50

model_mnist = keras.models.Sequential(name='ej4_mnist')
model_mnist.add(keras.layers.Input(shape=x_train[0].shape))
model_mnist.add(keras.layers.Conv2D(10,
    (24, 24),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same',
    name='filtro1'))
model_mnist.add(keras.layers.MaxPooling2D((2, 2),
    padding='same'))
model_mnist.add(keras.layers.Conv2D(10,
    (12, 12),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same',
    name='filtro2'))
model_mnist.add(keras.layers.Flatten())
model_mnist.add(keras.layers.Dense(10,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf),
    name='output'))

model_mnist.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
model_mnist.summary()

history = model_mnist.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=2)

model_mnist.save_weights('weights_ej5_b')
model_mnist.save('model_mnist.h5')
'''

model_mnist = keras.models.load_model('model_mnist.h5')

img_width = 28
img_height = 28

layer = model_mnist.get_layer(name='output')

feature_extractor_mnist = keras.Model(inputs=model_mnist.inputs, outputs=layer.output)

def compute_loss(input_image, index, layer):
    activation = feature_extractor_mnist(input_image)
    if layer=='filter':
        filter_activation = activation[:, 2:-2, 2:-2, index]
    elif layer=='class':
        filter_activation = activation[:, index]
    else:
        print('invalid class')
        return
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, index, learning_rate, layer, max_or_min):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, index, layer)

    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    if max_or_min=='max':
        img += learning_rate * grads
    elif max_or_min=='min':
        img -= learning_rate * grads
    else:
        print('invalid selection of learning')
        return
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 1))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(index, it=30, lr=10, layer='filter', max_or_min='max', type_img='noise', filename=None):
    # We run gradient ascent for 20 steps
    iterations = it
    learning_rate = lr
    if type_img=='noise':
        img = initialize_image()
    elif type_img=='specific':
        img = plt.imread(filename).astype(np.float32)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    else:
        print('invalida image type')
        return
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, index, learning_rate, layer, max_or_min)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[1:-1, 1:-1, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

fig = plt.figure(figsize=(19,3))
for i in range(10):
    loss, img = visualize_filter(i, 100, 1, 'class', 'max')
    ax = plt.subplot(1, 10, i+1)
    plt.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(i), fontsize=14, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Clases interpretadas por la red neuronal para MNIST.', fontsize=16)
plt.savefig('ej4_b.pdf', format='pdf')
plt.show()
