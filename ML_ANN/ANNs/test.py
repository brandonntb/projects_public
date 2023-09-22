from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Reshape, Flatten, BatchNormalization, Lambda, Rescaling
from keras.layers import LeakyReLU, Input, Dropout, Conv2DTranspose
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras import losses
from keras.models import load_model
from keras.regularizers import L1L2
import os
import re

import pickle
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))



try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.compat.v1.InteractiveSession(config=config)

except Exception as e:
    pass

def create_conv_vae(input_shape, latent_dim, dropout_rate, batch_size,
                start_lr=0.0001):
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    input_img = Input(shape=input_shape)

    x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(input_img, [z_mean, z_log_var, l], name="encoder")

    encoder.summary()

    z = Input(shape=(latent_dim,))

    x = LeakyReLU()(z)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 1))(x)
    x = apply_bn_and_dropout(x)
    x = Conv2DTranspose(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = apply_bn_and_dropout(x)
    x = Conv2DTranspose(128, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(1, (7, 7), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(input_shape[0] * input_shape[1] * input_shape[2], activation='sigmoid')(x)
    decoded = Reshape((input_shape[0], input_shape[1], input_shape[2]))(x)

    models["encoder"] = Model(input_img, l, name='Encoder')
    models["z_meaner"] = Model(input_img, z_mean, name='Enc_z_mean')
    models["z_lvarer"] = Model(input_img, z_log_var, name='Enc_z_log_var')

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, input_shape[0] * input_shape[1]*input_shape[2]))
        decoded = K.reshape(decoded, shape=(batch_size, input_shape[0] * input_shape[1]*input_shape[2]))
        xent_loss = input_shape[0] * input_shape[2] * input_shape[1]*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss)/2/input_shape[0]/input_shape[1]/input_shape[2]

    models["decoder"] = Model(z, decoded, name='Decoder')
    models["vae"] = Model(input_img,
                          models["decoder"](models["encoder"](input_img)),
                          name="VAE")

    models["vae"].compile(optimizer=Adam(lr=start_lr), loss=vae_loss, experimental_run_tf_function=False)

    return models, vae_loss


batch_size = 500
latentdim = 10

models,_ = create_conv_vae(
        input_shape=(28, 28, 1),
        latent_dim=latentdim,
        dropout_rate=0.3,
        batch_size=batch_size
    )

print(models)

vae = models["vae"]

models['decoder'].summary()

print(vae.summary())

vae.fit(x_train, x_train, batch_size=batch_size, verbose=1, epochs=20)

