import glob
import random
from textwrap import wrap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Rescaling, BatchNormalization, Dense, Conv2D, MaxPooling2D
from keras.preprocessing import image

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

import time

import winsound


#   creating a batch of images to show, consider altering so has dict and ds in arguments passed
def create_batch_fig(size, col, row, datasetX, datasetY):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("A Batch of Images Showing What the Model will be Trained on\n", fontsize=16)
    for i in range(size):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(datasetX[i])
        plt.xlabel(class_dict[datasetY[i]])

    fig.tight_layout()
    plt.show()


#   Architecture functions --------------------------------
def single_layer_conv(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def two_layer_conv(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def three_layer_conv(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def four_layer_BatchNorm(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))
    model.add(BatchNormalization())

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(BatchNormalization())

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(BatchNormalization())

    #   Layer 4
    model.add(Conv2D(256, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(BatchNormalization())

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def five_layer_basic(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 4
    model.add(Conv2D(256, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 5
    model.add(Conv2D(512, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def six_layer_basic(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 4
    model.add(Conv2D(256, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 5
    model.add(Conv2D(512, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Layer 5
    model.add(Conv2D(1024, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def four_layer_final(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))
    model.add(Dropout(0.3))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 4
    model.add(Conv2D(256, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def five_layer_final(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))
    model.add(Dropout(0.3))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 4
    model.add(Conv2D(256, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 5
    model.add(Conv2D(512, k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=p_size))
    model.add(Dropout(0.3))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def create_confusion_matrix(Y, Y_pred, class_dict):
    cm = confusion_matrix(Y.argmax(axis=1), Y_pred.argmax(axis=1))

    confM = sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_dict.values(),
                        yticklabels=class_dict.values())
    # confM.axis('scaled')
    plt.title('CNN Class Prediction Confusion Matrix')
    # plt.yticks(rotation=0)
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")

    plt.show()


def total_class_membership_plot(valdict, refdict, datasetY, dsname):
    print("-+-+-+-+-+-+-+MEMBERSHIP FOR THE {} DATASET:-+-+-+-+-+-+-+".format(dsname))
    for n in range(0, len(refdict)):
        for i in datasetY:
            if i == n:
                valdict[refdict.get(i)] += 1

    print("TOTAL In Dataset: {} Images\n".format(sum(valdict.values())))

    for c in valdict:
        print("{}: {} Images".format(c, valdict[c]))

    #   BAR
    plt.figure()
    plt.bar(list(refdict.values()), list(valdict.values()), color='royalblue')

    plt.title("Total Number of Classes in the {} Dataset".format(dsname))
    plt.xlabel("Classes")
    plt.ylabel("Number of Images in Each Class")

    plt.show()

    #   PIE

    colors = ['#004c6d', '#4c7c9b', '#86b0cc', '#c1e7ff']

    plt.pie(list(valdict.values()), colors=colors, labels=list(refdict.values()))

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.title("Class Distribution in the {} Dataset".format(dsname))
    plt.show()

    for key in valdict:
        valdict[key] = 0


#   plots the first 32 features from the feature maps
def plot_feature(feature_map, title):
    plt.figure(figsize=(30, 30))
    plt.suptitle(title, fontsize=60)

    row = 8
    count = 1
    for i in range(4):
        for n in range(row):
            plt.subplot(row, row, count)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(feature_map[0, :, :, count - 1])
            count += 1

    plt.tight_layout()
    plt.show()


def get_convolutional_layers(model, image):
    conv_layers = []
    feature_maps = []
    for layer in model.layers:
        if 'conv2d' in layer.name:
            layer_out = Model(inputs=model.inputs, outputs=layer.output)
            conv_layers.append(layer_out)
        else:
            continue

    for layer in conv_layers:
        feature_map = layer.predict(image)
        feature_maps.append(feature_map)

    for i in range(len(feature_maps)):
        plot_feature(feature_maps[i], "Convolutional Layer {} \n".format(i + 1))


#   ------------------------------------------------------------------------------------------------
#   predict class membership based on an image, using the trained model
def predictor(model, img_path):
    print("\n-+-+-+[IMG BEING ANALYSED]-+-+-+'")
    img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    plt.imshow(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    # print(prediction*100)
    # score = tf.nn.softmax(prediction[0])
    # print(score)

    for i in class_dict:
        if np.argmax(prediction) == i:
            plt.title("\n".join(wrap("This Image is Predicted to Belong to the {} Class with a Confidence of {:.2f} %."
                                     .format(class_dict[i], np.max(prediction) * 100))))

    plt.show()

    return img


def load_data(arg):
    if arg == 0:
        #  orig split
        trainX = pickle.load(open("trainX.pickle", "rb"))
        trainY = pickle.load(open("trainY.pickle", "rb"))

        testX = pickle.load(open("testX.pickle", "rb"))
        testY = pickle.load(open("testY.pickle", "rb"))

        valX = pickle.load(open("valX.pickle", "rb"))
        valY = pickle.load(open("valY.pickle", "rb"))

        class_dict = {0: "NORMAL", 1: "DRUSEN", 2: "DME", 3: "CNV"}
        number_in_class = {"NORMAL": 0, "DRUSEN": 0, "DME": 0, "CNV": 0}

        total_class_membership_plot(number_in_class, class_dict, trainY, "TRAIN")
        total_class_membership_plot(number_in_class, class_dict, testY, "TEST")
        total_class_membership_plot(number_in_class, class_dict, valY, "VAL")

    if arg == 2:
        #   ------------------------AUGMENTED DATA-----------------------------

        dataset = pickle.load(open("full_aug_datasetX.pickle", "rb"))
        labels = pickle.load(open("full_aug_datasetY.pickle", "rb"))

        class_dict = {0: "NORMAL", 1: "DRUSEN", 2: "DME", 3: "CNV"}
        number_in_class = {"NORMAL": 0, "DRUSEN": 0, "DME": 0, "CNV": 0}

        print("-+-+-+-FULL DATASET TOTAL CLASS MEMBERSHIP+-+-+-+-")
        print(dataset.shape)

        total_class_membership_plot(number_in_class, class_dict, labels, "full")

        trainX, testX, trainY, testY = train_test_split(dataset, labels, train_size=0.70, random_state=97)

        testX, valX, testY, valY = train_test_split(testX, testY, train_size=0.5, random_state=97)

        del testX, testY

        dataset = pickle.load(open("full_datasetX.pickle", "rb"))
        labels = pickle.load(open("full_datasetY.pickle", "rb"))

        nullX, testX, nullY, testY = train_test_split(dataset, labels, train_size=0.70, random_state=97)

        testX, null2X, testY, null2Y = train_test_split(testX, testY, train_size=0.5, random_state=97)

        total_class_membership_plot(number_in_class, class_dict, trainY, "TRAIN")
        total_class_membership_plot(number_in_class, class_dict, testY, "TEST")
        total_class_membership_plot(number_in_class, class_dict, valY, "VAL")

        del dataset, labels, nullX, null2X

    if arg == 3:
        dataset = pickle.load(open("full_datasetX.pickle", "rb"))
        labels = pickle.load(open("full_datasetY.pickle", "rb"))

        class_dict = {0: "NORMAL", 1: "DRUSEN", 2: "DME", 3: "CNV"}
        number_in_class = {"NORMAL": 0, "DRUSEN": 0, "DME": 0, "CNV": 0}

        print("-+-+-+-FULL DATASET TOTAL CLASS MEMBERSHIP+-+-+-+-")
        print(dataset.shape)

        total_class_membership_plot(number_in_class, class_dict, labels, "full")

        trainX, testX, trainY, testY = train_test_split(dataset, labels, train_size=0.70, random_state=97)

        testX, valX, testY, valY = train_test_split(testX, testY, train_size=0.5, random_state=97)

        total_class_membership_plot(number_in_class, class_dict, trainY, "TRAIN")
        total_class_membership_plot(number_in_class, class_dict, testY, "TEST")
        total_class_membership_plot(number_in_class, class_dict, valY, "VAL")

        del dataset, labels

    return trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class


#   ---------------MEMORY RESTRICTION TO PREVENT CRASHES----------------

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.compat.v1.InteractiveSession(config=config)
except Exception as e:
    pass

#   -----------------------------------------------------------------------------------
print('\n-+-+-+[SID: 8726104]-+-+-+')
#   ------------------------FILE / DATASET OPTIONS-------------------------------------

print('\n-+-+-+[Grabbing Files For Predictions]-+-+-+')
# Collecting some images for prediction later on
pred_base = glob.glob("../Dataset - OCT/OCT2017/test/*")
pred_imgs = []

for path in pred_base:
    for i in range(1):
        pred_imgs.append(random.choice(glob.glob(path + "/*.jpeg")))

print('\n-+-+-+[Prediction Files are:]-+-+-+')
print(pred_imgs)

print('\nLoading Files...')

#   NORMAL dataset
# trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class = load_data(0)

#   ANOMALY (2 class) DATSET
# trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class = load_data(1)

#   AUGMENTED DATASET
trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class = load_data(2)

# #   FULL DATASET
# trainX, trainY, testX, testY, valX, valY, class_dict, number_in_class = load_data(3)


#   --------------------------------- OVERFITTING MEASURES ------------------------------------------

early_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, mode="min", restore_best_weights=True,
                               patience=5, verbose=1)
reduce_lrn = ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1, factor=0.3, min_lr=0.001)

callbacks = [early_callback, reduce_lrn]

# weight_shift = compute_class_weight('balanced', np.unique(trainY), np.asarray(trainY))
# class_weights = dict(zip(np.unique(trainY), weight_shift))

# print(class_weights)

#   --------------------------------------------------------------------------------------------------

create_batch_fig(9, 3, 3, trainX, trainY)

trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)
valY = to_categorical(valY, num_classes=4)

print('\n-+-+-+[Files Loaded]-+-+-+')

print('\nCreating Model...')

model = four_layer_final(trainX)

print('\n-+-+-+[Model Created]-+-+-+')


print('\nTraining Model...')
start = time.perf_counter()

observed = model.fit(trainX, trainY, batch_size=64, epochs=20, validation_data=(valX, valY)
                     , shuffle=True, callbacks=callbacks)

# observed = model.fit(trainX, trainY, batch_size=64, epochs=20, validation_data=(valX, valY)
#                       , shuffle=True, callbacks=callbacks, class_weight=class_weights)


end = time.perf_counter()
print('\n-+-+-+[Model Trained in {:.2f}s]-+-+-+'.format(end - start))

#  below is all the val and train vals plotted on one graph
pd.DataFrame(observed.history).plot()
plt.title('Model Accuracies and Losses')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

print('\nTesting the Accuracy of the Model...')
test_model = model.evaluate(testX, testY, verbose=1)
print('\nTest Score = ', test_model[0])
print('Test Accuracy = ', test_model[1] * 100, "%")

print('\n-+-+-+[Finished !!]-+-+-+')
#
#
print('\n-+-+-+[Starting Visual Analysis]-+-+-+')

# VISUAL ANALYSIS ----------------------------------->>>>>>_______________

for img in pred_imgs:
    get_convolutional_layers(model, predictor(model, img))

# for img in pred_imgs:
#     predictor(model, img)


print("\n-+-+-+[TESTING]-+-+-+'")

test_predictions = model.predict(testX, verbose=1)
print(
    "\nPrediction Accuracy is : {}".format(accuracy_score(testY.argmax(axis=1), test_predictions.argmax(axis=1)) * 100))

create_confusion_matrix(testY, test_predictions, class_dict)

print("")
print(classification_report(y_true=testY.argmax(axis=1), y_pred=test_predictions.argmax(axis=1),
                            target_names=class_dict.values()))

print('\n-+-+-+[SID: 8726104]-+-+-+')

# Notification for when testing finished
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
