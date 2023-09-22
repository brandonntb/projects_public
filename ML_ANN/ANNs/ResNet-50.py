import pickle
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from keras.models import Sequential, Model
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, GlobalAvgPool2D, Dropout
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time
import winsound


def load_data():
    dataset = pickle.load(open("full_datasetX.pickle", "rb"))
    labels = pickle.load(open("full_datasetY.pickle", "rb"))

    class_dict = {0: "NORMAL", 1: "DRUSEN", 2: "DME", 3: "CNV"}
    number_in_class = {"NORMAL": 0, "DRUSEN": 0, "DME": 0, "CNV": 0}

    print("-+-+-+-FULL DATASET TOTAL CLASS MEMBERSHIP+-+-+-+-")
    print(dataset.shape)

    trainX, testX, trainY, testY = train_test_split(dataset, labels, train_size=0.70, random_state=97)

    testX, valX, testY, valY = train_test_split(testX, testY, train_size=0.5, random_state=97)

    del dataset, labels

    return trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class


def ResNet50_model_additional_layers(ds):
    input = ds.shape[1:]
    print(input)
    classes = 4

    resnet = ResNet50(include_top=False, input_shape=input, pooling='avg', weights="imagenet")
    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resnet)

    model.add(Flatten())

    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def ResNet50_model(ds):
    input = ds.shape[1:]
    print(input)
    classes = 4

    resnet = ResNet50(include_top=False, input_shape=input, pooling='avg', weights="imagenet")
    for layer in resnet.layers[:-14]:
        layer.trainable = False

    model = Sequential()
    model.add(resnet)

    model.add(Flatten())

    #     model.add(Dense(100, activation ='relu'))
    #     model.add(Dense(100, activation ='relu'))

    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_confusion_matrix(Y, Y_pred):
    cm = confusion_matrix(Y.argmax(axis=1), Y_pred.argmax(axis=1))

    confM = sns.heatmap(cm, annot=True, fmt='g', cmap='magma', xticklabels=class_dict.values(),
                        yticklabels=class_dict.values())
    confM.axis('scaled')
    plt.title('Class Prediction Confusion Matrix')
    plt.yticks(rotation=0)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.show()


#   ---------------IN CASE OF CRASHES OR RUNNING OUT OR MEMORY USE THESE----------------

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.compat.v1.InteractiveSession(config=config)

#   --------------------------------------------------------------------------------------
print('\n-+-+-+[SID: 8726104]-+-+-+')

print('\nLoading Files...')
trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class = load_data()

trainY = to_categorical(np.asarray(trainY), num_classes=4)
testY = to_categorical(np.asarray(testY), num_classes=4)
valY = to_categorical(np.asarray(valY), num_classes=4)
print('\n-+-+-+[Files Loaded]-+-+-+')

#   ---------------------------TEST --------------------------------------------------

print(trainX.shape)
trainX = np.repeat(trainX, 3, -1)
testX = np.repeat(testX, 3, -1)
valX = np.repeat(valX, 3, -1)
print(trainX.shape)

#   -----------------------------------------------------------------------------------


early_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, mode="min", restore_best_weights=True,
                               patience=5, verbose=1)
reduce_lrn = ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1, factor=0.3, min_lr=0.001)

callbacks = [early_callback, reduce_lrn]

model = ResNet50_model(trainX)

model.summary()

start = time.perf_counter()
observed = model.fit(trainX, trainY, batch_size=32, epochs=20, validation_data=(valX, valY)
                     , shuffle=True, callbacks=callbacks)

end = time.perf_counter()
print('\n-+-+-+[Model Trained in {:.2f}s]-+-+-+'.format(end - start))

pd.DataFrame(observed.history).plot()
plt.title('Model Accuracies and Losses')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

print('\nTesting the Accuracy of the Model...')
test_model = model.evaluate(testX, testY, verbose=1)
print('\nTest Score = ', test_model[0])
print('Test Accuracy = ', test_model[1] * 100, "%")

test_predictions = model.predict(testX, verbose=1)
print(
    "\nPrediction Accuracy is : {}".format(accuracy_score(testY.argmax(axis=1), test_predictions.argmax(axis=1)) * 100))

create_confusion_matrix(testY, test_predictions)

print("")
print(classification_report(y_true=testY.argmax(axis=1), y_pred=test_predictions.argmax(axis=1),
                            target_names=class_dict.values()))

print('\n-+-+-+[SID: 8726104]-+-+-+')

