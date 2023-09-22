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

import tensorflow as tf

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.python.framework.ops import disable_eager_execution

import pickle

from sklearn_som.som import SOM
from minisom import MiniSom

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



from keras.losses import mean_squared_error, mae

import tensorflow as tf
from tensorflow import math

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

disable_eager_execution()

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.compat.v1.InteractiveSession(config=config)

except Exception as e:
    pass

def create_conv_vae(input_shape, latent_dim, dropout_rate,
                start_lr=0.0001):
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    input_img = Input(shape=input_shape)
    x = Rescaling(1./255)(input_img)
    x = Conv2D(128, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    batch_size = tf.shape(z_mean)[0]

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    lambda_layer = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

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

    models["encoder"] = Model(input_img, lambda_layer,name='Encoder')
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



train = pickle.load(open("autoenc_train.pickle", "rb"))

# train = train.astype('float32')/255

testX = pickle.load(open("testX.pickle", "rb"))

# testX = testX.astype('float32')/255

testY = pickle.load(open("testY.pickle", "rb"))

val = pickle.load(open("autoenc_val.pickle", "rb"))
valY = pickle.load(open("valY.pickle", "rb"))

# val = val.astype('float32')/255

latent_dim = 10

models,_ = create_conv_vae(
        input_shape=(128, 128, 1),
        latent_dim=latent_dim,
        dropout_rate=0.3,
    )


vae = models["vae"]

models['encoder'].summary()
models['decoder'].summary()

print(vae.summary())

vae.fit(testX, testX, batch_size=16, verbose=1, epochs=5)


# from tensorflow.keras.utils import plot_model
# plot_model(models['decoder'], show_shapes=True)

X_encode = models['encoder'].predict(val)

data = X_encode

# plt.figure(figsize=(12,10))
# plt.scatter(data[:, 0], data[:, 1], c=testY,
#             edgecolor='none', alpha=0.7, s=40,
#             cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.title('images projection')
# plt.show()

features = np.reshape(X_encode, newshape=(X_encode.shape[0], -1))

print(features.shape)

x = 1
y = 4
som = MiniSom(x, y, input_len=features.shape[1], sigma=0.5, learning_rate=.5,
              neighborhood_function='gaussian')


som.train(features, 1000,verbose=True)


# each neuron represents a cluster
winner_coordinates = np.array([som.winner(f) for f in features]).T

# results = np.ravel_multi_index(winner_coordinates, (x,y))
s_results = [i for i in winner_coordinates[1]]

hit_miss = {"hit":0, "miss":0}

# for i in (range(len(valY))):
#     if s_results[i] == valY[i]:
#         hit_miss["hit"] += 1
#     else:
#         hit_miss["miss"] += 1
#
# print(hit_miss)

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(s_results):
    plt.scatter(features[s_results == c, 0],
                features[s_results == c, 1], label='cluster='+str(c))

for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=2, linewidths=12, color='k', label='centroid')

plt.legend()
plt.title('SOM')
plt.show()
