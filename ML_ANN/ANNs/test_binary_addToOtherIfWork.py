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

import matplotlib.cm as cm
from IPython.display import Image, display

from scipy import ndimage


#   Architecture functions --------------------------------

def four_layer_model():
    k_size = (3, 3)
    p_size = (2, 2)
    classes = 2

    model = Sequential()
    #   dividing the pixels in the image by the total range (255) so either 1 (white) or 0 (black)
    model.add(Rescaling(1. / 255, input_shape=(128, 128, 1)))

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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
            plt.title("\n".join(wrap("This Image is Predicted to contain {} data with a Confidence of {:.2f} %."
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

trainX = pickle.load(open("anomTrainX.pickle", "rb"))
trainY = pickle.load(open("anomTrainy.pickle", "rb"))

testX = pickle.load(open("anomTestX.pickle", "rb"))
testY = pickle.load(open("anomTesty.pickle", "rb"))

valX = pickle.load(open("anomValX.pickle", "rb"))
valY = pickle.load(open("anomValy.pickle", "rb"))

class_dict = {0: "NORMAL", 1: "ANOMALY"}

early_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, mode="min", restore_best_weights=True,
                               patience=5, verbose=1)
reduce_lrn = ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1, factor=0.3, min_lr=0.001)

callbacks = [early_callback, reduce_lrn]

#   --------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>


trainY = to_categorical(np.asarray(trainY), num_classes=2)
valY = to_categorical(np.asarray(valY), num_classes=2)
testY = to_categorical(np.asarray(testY), num_classes=2)

print('\n-+-+-+[Files Loaded]-+-+-+')

print('\nCreating Model...')
model = four_layer_model()

print('\n-+-+-+[Model Created]-+-+-+')

print('\nTraining Model...')

observed = model.fit(trainX, trainY, batch_size=32, validation_data=(valX, valY)
                     , epochs=1, shuffle=True, callbacks=callbacks)


# print('\n-+-+-+[Model Trained]-+-+-+')
#
# pd.DataFrame(observed.history).plot()
# plt.title('Model Accuracies and Losses')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.show()
#
# print('\nTesting the Accuracy of the Model...')
# test_model = model.evaluate(testX, testY, verbose=1)
# print('\nTest Score = ', test_model[0])
# print('Test Accuracy = ', test_model[1] * 100, "%")
#
# print('\n-+-+-+[Finished !!]-+-+-+')
# #
# #
# print('\n-+-+-+[Starting Visual Analysis]-+-+-+')
#
# for img in pred_imgs:
#     predictor(model, img)
#
# print("\n-+-+-+[TESTING]-+-+-+'")
#
# test_predictions = model.predict(testX, verbose=1)
# print(
#     "\nPrediction Accuracy is : {}".format(accuracy_score(testY.argmax(axis=1), test_predictions.argmax(axis=1)) * 100))
#
# create_confusion_matrix(testY, test_predictions, class_dict)
#
# print("")
# print(classification_report(y_true=testY.argmax(axis=1), y_pred=test_predictions.argmax(axis=1),
#                             target_names=class_dict.values()))
#
#
#
