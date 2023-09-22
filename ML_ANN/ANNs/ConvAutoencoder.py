import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Rescaling, Reshape, Conv2DTranspose
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import mean_squared_error, mae

import tensorflow as tf
from tensorflow import math

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score

import pickle
import time



from sklearn.model_selection import train_test_split

#   ---------------IN CASE OF CRASHES OR RUNNING OUT OR MEMORY USE THESE----------------
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.compat.v1.InteractiveSession(config=config)

except Exception as e:
    pass


#   --------------------------------------------------------------------------------------

# #   --------------------------------------------- load data, change if it works---------------

def threshold_finder(predictions, actual):
    limits = [int(np.min(predictions)), int(np.max(predictions))]

    print(limits)

    precisions = []
    f1s = []
    recalls = []
    accuracies = []
    thresholds = []

    upper_limit = int(limits[1] / 2)
    lower_limit = int(limits[0] * 2)

    print("UPPER_LIMIT: {}\nLOWER_LIMIT: {}".format(upper_limit, lower_limit))

    for i in range(lower_limit, upper_limit, 100):
        threshold = i

        #         print(threshold)

        thresholds.append(threshold)

        y_pred = [1 if pred > threshold else 0 for pred in predictions]

        accuracies.append(accuracy_score(actual, y_pred))
        precisions.append(precision_score(actual, y_pred))
        recalls.append(recall_score(actual, y_pred))
        f1s.append(f1_score(actual, y_pred))

    best_accuracy = accuracies.index(max(accuracies))
    best_precision = precisions.index(max(precisions))

    #     print("Best Threshold: {} , yields an accuracy score of: {}".format(thresholds[best_accuracy]
    #                                                                         , accuracies[best_accuracy]))
    print(
        "Best Threshold: {} , yields a recall score of: {}".format(thresholds[best_precision], recalls[best_precision]))

    #     return thresholds[best_accuracy]
    return thresholds[best_precision]


def load_data():
    #   ------------------------ANOM DATA-----------------------------
    dataset = pickle.load(open("full_datasetX.pickle", "rb"))
    labels = pickle.load(open("full_datasetY.pickle", "rb"))

    print("-+-+-+-FULL DATASET TOTAL CLASS MEMBERSHIP+-+-+-+-")
    print(dataset.shape)

    trainX, testX, trainY, testY = train_test_split(dataset, labels, train_size=0.70, random_state=97)

    testX, valX, testY, valY = train_test_split(testX, testY, train_size=0.5, random_state=97)

    normal_idx = np.where(trainY == 0)[0]
    normal_val_idx = np.where(valY == 0)[0]

    trainX = np.asarray([trainX[i] for i in normal_idx])
    valX = np.asarray([valX[i] for i in normal_val_idx])

    anom_test_idx = np.where(testY != 0)[0]

    testY[anom_test_idx[:int(len(anom_test_idx) / 2) - 1000]] = 1

    rem = np.where(testY <= 1)[0]

    temp = []
    tempY = []
    for i in rem:
        temp.append(testX[i])
        tempY.append(testY[i])

    testX = np.asarray(temp)
    testY = np.asarray(tempY)

    print(testY)

    return trainX, trainY, testX, testY, valX, valY


def model():
    # consider testing conv2d transpose for the upsampling + conv layers

    input = (128, 128, 1)
    model = Sequential()

    k_size = (3, 3)
    p_size = (2, 2)

    model.add(Rescaling(1. / 255, input_shape=input))

    model.add(Conv2D(128, k_size, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=p_size))

    #   -----------------------------------
    model.add(Conv2D(64, k_size, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=p_size))

    model.add(Conv2D(32, k_size, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=p_size))

    #   -------------------------------------------------->>>

    model.add(Flatten())

    model.add(Dense(4, activation='sigmoid'))

    model.add(Dense(8192, activation='sigmoid'))

    model.add(Reshape((16, 16, 32)))

    #   -------------------------------------------------->>>

    model.add(BatchNormalization())
    model.add(Conv2D(32, k_size, padding='same', activation='relu'))

    model.add(UpSampling2D(p_size))

    model.add(Conv2D(64, k_size, padding='same', activation='relu'))

    model.add(UpSampling2D(p_size))
    #   --------------------------------------

    model.add(Conv2D(128, k_size, padding='same', activation='relu'))

    model.add(UpSampling2D(p_size))

    model.add(Conv2D(2, k_size, activation='sigmoid', padding='same'))

    # -------------------------

    # -------------------------

    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])
    return model


model = model()

trainX, trainY, testX, testY, valX, valY = load_data()

trainX = trainX[:1000]

print(trainX.shape)

model_path = "models/anom.ckpt"

save = ModelCheckpoint(monitor='mse', verbose=1, save_best_only=True,
                       save_weights_only=True, mode='min', save_freq='epoch', filepath=model_path)

reduce_lrn = ReduceLROnPlateau(monitor='mse', factor=0.4, patience=2, verbose=1, mode='max', min_delta=0.001)

#   val loss was for all these checkpoint things
early = EarlyStopping(monitor='mse', mode='min', patience=5, verbose=1, min_delta=0.001)

callbacks = [save, reduce_lrn, early]

# model.load_weights(model_path)

start = time.perf_counter()
# model.fit(trainX, trainX, validation_data=(valX, valX), epochs=20,
#           callbacks=callbacks, verbose=1, batch_size=64)
end = time.perf_counter()
print('\n-+-+-+[Model Trained in {:.2f}]-+-+-+'.format(end - start))
#
# #   ------------------------------------------------------------------------------------------------------------------
test_predictions = model.predict(testX, verbose=1)

#
predictions = []

for i in (mean_squared_error(testX, test_predictions)):
    # getting the mean of the errors for each pixel, for each image
    predictions.append(math.reduce_mean(i))

results = []
start = time.perf_counter()

threshold = threshold_finder(predictions, testY)

end = time.perf_counter()

print('\n-+-+-+[Threshold found in {:.2f}]-+-+-+'.format(end - start))

for i in predictions:
    if i > threshold:
        results.append(1)
    else:
        results.append(0)


confM = confusion_matrix(testY, results)

sns.heatmap(confM, annot=True, fmt='g', xticklabels=["NORMAL", "ANOMALY"]
            , yticklabels=["NORMAL", "ANOMALY"], cmap='magma')
plt.title('Autoencoder Anomaly Detection Confusion Matrix')
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.show()

print(classification_report(y_true=testY, y_pred=results, target_names=["NORMAL", "ANOMALY"]))

print("Accuracy = {}".format(accuracy_score(testY, results)))
print("Precision = {}".format(precision_score(testY, results)))
print("Recall = {}".format(recall_score(testY, results)))
