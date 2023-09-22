import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D, BatchNormalization,Reshape
import pickle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import time

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.compat.v1.InteractiveSession(config=config)
except Exception as e:
    pass


def SOM_model():
    # consider testing conv2d transpose for the upsampling + conv layers

    input = (128, 128, 1)
    model = Sequential()

    k_size = (3, 3)
    p_size = (2, 2)

    model.add(Rescaling(1. / 255, input_shape=input))

    model.add(Conv2D(128, k_size, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=p_size))
    #   -----------------------------------
    model.add(Conv2D(64, k_size, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=p_size))

    model.add(Conv2D(32, k_size, padding='same', activation='relu'))
    #   remove this (below) to see if performance effected research says only between conv layers)
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=p_size))

    #   -------------------------------------------------->>>

    model.add(Flatten())

    model.add(Dense(4, activation='sigmoid', name="for_cluster"))

    model.add(Dense(8192, activation='sigmoid'))

    model.add(Reshape((16, 16, 32)))

    #   -------------------------------------------------->>>

    model.add(Conv2D(32, k_size, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(p_size))

    model.add(Conv2D(64, k_size, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(p_size))
    #   --------------------------------------
    model.add(Conv2D(128, k_size, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(p_size))

    model.add(Conv2D(1, k_size, activation='sigmoid', padding='same'))

    # -------------------------

    # -------------------------

    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])
    return model



model = SOM_model()

model_path = "models/CAE_SOM.ckpt"

# model.load_weights(model_path)

trainX = pickle.load(open("trainX.pickle", "rb"))
trainY = pickle.load(open("trainY.pickle", "rb"))

trainX = trainX[:300]
trainY = trainY[:300]

testX = pickle.load(open("testX.pickle", "rb"))
valX = pickle.load(open("valX.pickle", "rb"))

testY = pickle.load(open("testY.pickle", "rb"))
valY = pickle.load(open("valY.pickle", "rb"))



save = ModelCheckpoint(monitor='mse', verbose=1, save_best_only=True,
                        save_weights_only=True, mode='min', save_freq='epoch', filepath=model_path)

reduce_lrn = ReduceLROnPlateau(monitor='mse', factor=0.4, patience=2, verbose=1, mode='max', min_delta=0.0001)

early = EarlyStopping(monitor='mse', mode='min', patience=10, verbose=1, min_delta=0.0001)

callbacks = [save, reduce_lrn, early]

model.fit(trainX, trainX, epochs=20, validation_data=(valX, valX), callbacks=callbacks)
# model.fit(valX, valX, epochs=20)


feature_model = Model(inputs=model.input, outputs=model.get_layer(name="for_cluster").output)

features = feature_model.predict(testX)

print(features.shape)

features2 = np.reshape(features, newshape=(features.shape[0], -1))

print(features.shape)

# #  ---------------------------------------------------------------------------------------

pca = PCA(2)

df = pca.fit_transform(features2)

km = KMeans(n_clusters=4)
k_results = km.fit_predict(df)

print(k_results)

uniq_k_results = np.unique(k_results)

for i in uniq_k_results:
    plt.scatter(df[k_results == i , 0] , df[k_results == i , 1] , label = i)
plt.legend()
plt.show()

hit_miss = {"hit":0, "miss":0}

for i in (range(len(testY))):
    if k_results[i] == testY[i]:
        hit_miss["hit"] += 1
    else:
        hit_miss["miss"] += 1

print(hit_miss)

print(classification_report(y_true=testY, y_pred=k_results, target_names=["NORMAL", "DRUSEN", "DME", "CNV"]))

#   -----------------------------------------------------------
# x, y are dimensions of som, input len is number of elements going into the som
# to sigma and neighbourhood func are how winnsers are determined and
# the amount the winnder and neighbours move by when winner selected

print(features.shape)

x = 1
y = 4
som = MiniSom(x, y, input_len=features.shape[1], sigma=1.3, learning_rate= 0.5,
              neighborhood_function='triangle')

start = time.perf_counter()

som.train(features, 100000,verbose=True)

end = time.perf_counter()
print("time taken: {}s".format(end-start))

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(f) for f in features]).T


s_results = [i for i in winner_coordinates[1]]

hit_miss = {"hit":0, "miss":0}

for i in (range(len(testY))):
    if s_results[i] == testY[i]:
        hit_miss["hit"] += 1
    else:
        hit_miss["miss"] += 1

print(hit_miss)

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(s_results):
    plt.scatter(features[s_results == c, 0],
                features[s_results == c, 1], label='cluster='+str(c))

for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=2, linewidths=12, color='k', label='centroid')

plt.legend()
plt.title('SOM Accuracy = {}'.format(accuracy_score(testY, s_results)))
plt.show()

print("Accuracy = {}".format(accuracy_score(testY, s_results)))


