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

# -----------------------> new testing

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import warnings


#   FUNCTIONS STABLE -------------------->>>>CONSIDER MOVING VIS ONES TO SEPARATE FILE FOR IMPORT
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

def four_layer_model(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)
    classes = 4

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=p_size))
    model.add(Dropout(0.3))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 4
    model.add(Conv2D(128, k_size, activation='relu'))
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())
    # number of last dense should be the number of classes
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def Four_layerBatchNorm_model(ds):
    input = ds.shape[1:]

    k_size = (3, 3)
    p_size = (2, 2)

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=input))

    #   Layer 1
    model.add(Conv2D(32, k_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=p_size))
    model.add(Dropout(0.3))

    #   Layer 2
    model.add(Conv2D(64, k_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 3
    model.add(Conv2D(128, k_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Layer 4
    model.add(Conv2D(128, k_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(p_size))
    model.add(Dropout(0.3))

    #   Last / Final fully-connected Layer (5)
    model.add(Flatten())
    # number of last dense should be the number of classes
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

#   -------------------------------------------------------------------------------

def create_confusion_matrix(Y, Y_pred, class_dict):
    cm = confusion_matrix(Y.argmax(axis=1), Y_pred.argmax(axis=1))

    confM = sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_dict.values(),
                        yticklabels=class_dict.values())
    # confM.axis('scaled')
    plt.title('Class Prediction Confusion Matrix')
    # plt.yticks(rotation=0)
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")

    plt.show()


def total_class_membership_plot(valdict, refdict, datasetY):
    for n in range(0, len(refdict)):
        for i in datasetY:
            if i == n:
                valdict[refdict.get(i)] += 1

    for c in valdict:
        print("{}: {} Images".format(c, valdict[c]))

    plt.figure()
    plt.bar(list(refdict.values()), list(valdict.values()), color='royalblue')

    plt.title("Total Number of Classes in the Dataset")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images in Each Class")

    plt.show()


#   get the CNN layers in the model
def get_convolutional_layers(model):
    convolution_layers = []
    for layer in model.layers:
        if 'conv2d' not in layer.name:
            continue
        model_temp = Model(inputs=model.inputs, outputs=layer.output)
        convolution_layers.append(model_temp)
    return convolution_layers


#   get feature maps for each level
def fetch_feature_maps(model, test_image):
    models = get_convolutional_layers(model)  # Fetching convolution layers models
    feature_maps = []

    for model_temp in models:
        feature_map = model_temp.predict(test_image)
        feature_maps.append(feature_map)
    return feature_maps, models


#   Plot the 32 maps in 8 by 4
def plot_feature(feature_map, title):
    fig = plt.figure(figsize=(30, 30))
    fig.suptitle(title, fontsize=40)

    dim = 8
    idx = 1
    for _ in range(4):
        for _ in range(8):
            ax = plt.subplot(dim, dim, idx)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_map[0, :, :, idx - 1])
            idx += 1

    fig.tight_layout()
    plt.show()


#   generate feature maps based on an image and the trained model
def feature_generator(model, img):
    feature_maps, models = fetch_feature_maps(model, img)
    counter = 1
    for each_map in feature_maps:
        plot_feature(each_map, "Convolutional Layer {} \n ".format(counter))
        counter += 1


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

#   ---------------IN CASE OF CRASHES OR RUNNING OUT OR MEMORY USE THESE----------------

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # session = tf.compat.v1.InteractiveSession(config=config)
except Exception as e:
    pass

#   --------------------------------------------------------------------------------------

#   FUNCTIONS test ----------------------------------------------------------->

# div by 255 to make each pixel value be either a 0 (black) or 1 (white) (range of pixels reduced to binary)
# trainX = trainX / 255
# testX = testX / 255
# valX = valX / 255
#   test above against resampling/ shaping (used in layers) pretty sure layers is better

print('\n-+-+-+[Grabbing Files For Predictions]-+-+-+')
# Collecting some images for prediction later on
pred_imgs = glob.glob("../Dataset - OCT/Prediction tests/*")
all_test_paths = glob.glob("../Dataset - OCT/OCT2017/test/*/")

for path in all_test_paths:
    for i in range(1):
        pred_imgs.append(random.choice(glob.glob(path + "*.jpeg")))

print('\n-+-+-+[Prediction Files are:]-+-+-+')
print(pred_imgs)

print('\nLoading Files...')
trainX = pickle.load(open("training_dataX.pickle", "rb"))
trainY = pickle.load(open("training_datay.pickle", "rb"))

testX = pickle.load(open("test_dataX.pickle", "rb"))
testY = pickle.load(open("test_datay.pickle", "rb"))

valX = pickle.load(open("validation_dataX.pickle", "rb"))
valY = pickle.load(open("validation_datay.pickle", "rb"))

# aug_trainX = pickle.load(open("augmented_dataX.pickle", "rb"))
# aug_trainY = pickle.load(open("augmented_datay.pickle", "rb"))

class_dict = {0: "NORMAL", 1: "CNV", 2: "DME", 3: "DRUSEN"}

number_in_class = {"NORMAL": 0, "CNV": 0, "DME": 0, "DRUSEN": 0}

total_class_membership_plot(number_in_class, class_dict, trainY)

#   TESTING ----------------------------------------------------------->
# ------------ seem to make acc worse?? continue testing with diff layers

early_callback = EarlyStopping(monitor="val_loss", min_delta=0, mode="min", restore_best_weights=True,
                               patience=5, verbose=1)
reduce_lrn = ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1, factor=0.3, min_lr=0.001)

callbacks = [early_callback, reduce_lrn]


# # weight_shift = compute_class_weight('balanced', np.unique(trainY), np.asarray(trainY))
# # class_weights = dict(zip(np.unique(trainY), weight_shift))

# # print(class_weights)

#   --------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>

create_batch_fig(9, 3, 3, trainX, trainY)

trainY = to_categorical(np.asarray(trainY), num_classes=4)
testY = to_categorical(np.asarray(testY), num_classes=4)
valY = to_categorical(np.asarray(valY), num_classes=4)

# aug_trainY = to_categorical(np.asarray(aug_trainY), num_classes=4)
print('\n-+-+-+[Files Loaded]-+-+-+')

print('\nCreating Model...')
model = four_layer_model(trainX)
# model = Four_layerBatchNorm_model(atrainX)
# model = four_layer_model(aug_trainX)
print('\n-+-+-+[Model Created]-+-+-+')

#

# plot_model(model,show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True,
#     to_file='model.png')
#
#
print('\nTraining Model...')
# observed = model.fit(trainX, trainY, batch_size=32, epochs=1, validation_data=(valX, valY)
#                       , shuffle=True)

observed = model.fit(trainX, trainY, batch_size=32, epochs=20, validation_data=(valX, valY)
                      , shuffle=True, callbacks=callbacks)

# observed = model.fit(aug_trainX, aug_trainY, batch_size=32, epochs=20, validation_data=(valX, valY)
#                       , shuffle=True)
print('\n-+-+-+[Model Trained]-+-+-+')

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
print('\n-+-+-+[Starting Visual Analysis]-+-+-+')

# VISUAL ANALYSIS ----------------------------------->>>>>>

# currently random ones from the internet, change to some from test set before submission
# img_path = "../Dataset - OCT/OCT2017/test/DRUSEN/DRUSEN-1219727-1.jpeg"

# for img in pred_imgs:
#     feature_generator(model, predictor(model, img))

for img in pred_imgs:
    predictor(model, img)


print("\n-+-+-+[TESTING]-+-+-+'")

test_predictions = model.predict(testX, verbose=1)
print(
    "\nPrediction Accuracy is : {}".format(accuracy_score(testY.argmax(axis=1), test_predictions.argmax(axis=1)) * 100))

create_confusion_matrix(testY, test_predictions, class_dict)

print("")
print(classification_report(y_true=testY.argmax(axis=1), y_pred=test_predictions.argmax(axis=1),
                            target_names=class_dict.values()))

