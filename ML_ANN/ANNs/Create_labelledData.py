# Loading in my own data
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
from tqdm import tqdm
import glob


def create_array(path, category, categories, dataset):
    if len(categories) > 2:
        class_num = categories.index(category)
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

            # re-sizing an image resolution for standardisation
            resized_array = cv2.resize(img_array, (img_size, img_size))
            dataset.append([resized_array, class_num])

    else:
        if category == "NORMAL":
            class_num = 0
        else:
            class_num = 1

        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

            # re-sizing an image resolution for standardisation
            resized_array = cv2.resize(img_array, (img_size, img_size))
            dataset.append([resized_array, class_num])


def create_dataset(export_name,categories, base_path, dataset):
    for cat in class_categories:
        path = os.path.join(base_path, cat).replace("\\", "/")  # path to the category folders

        create_array(path, cat, categories, dataset)

    print("Shuffling...")
    random.shuffle(dataset)

    # X is feature set, y is label set
    X_set = []
    Y_set = []


    for features, label in tqdm(dataset):
        X_set.append(features)
        Y_set.append(label)

    img_set = np.array(X_set).reshape(-1, img_size, img_size, 1)
    label_setY = np.array(Y_set)

    pickle_out = open(export_name + "X.pickle", "wb")
    pickle.dump(img_set, pickle_out)
    pickle_out.close()

    pickle_out = open(export_name + "Y.pickle", "wb")
    pickle.dump(label_setY, pickle_out)
    pickle_out.close()


def dataset_combiner(datasetsX, datasetsY, name):
    dataset = np.concatenate((datasetsX))
    labels = np.concatenate((datasetsY))

    print("_+_+_+_+DATASET_+__+_+_+_+_",dataset.shape)
    print("_+_+_+_+LABELS_+__+_+_+_+_",labels.shape)

    pickle_out = open(name + "X" + ".pickle", "wb")
    pickle.dump(dataset, pickle_out)
    pickle_out.close()

    pickle_out = open(name + "Y" + ".pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()


train_path = "../Dataset - OCT/OCT2017/train"
test_path = "../Dataset - OCT/OCT2017/test"
validation_path = "../Dataset - OCT/OCT2017/val"

#
img_size = 128
#

class_categories = os.listdir(train_path)
class_categories.reverse()
print(class_categories)


train_data = []
test_data = []
val_data = []


print("Creating and Exporting Datasets...")
create_dataset("train",class_categories, train_path, train_data)
create_dataset("val",class_categories, validation_path, val_data)
create_dataset("test",class_categories, test_path, test_data)


#   -----------------------------FULL DS DATAGEN----------------------------------------------

trainX = pickle.load(open("trainX.pickle", "rb"))
trainY = pickle.load(open("trainY.pickle", "rb"))

testX = pickle.load(open("testX.pickle", "rb"))
testY = pickle.load(open("testY.pickle", "rb"))

valX = pickle.load(open("valX.pickle", "rb"))
valY = pickle.load(open("valY.pickle", "rb"))
#

dataset_combiner([trainX, testX, valX], [trainY, testY, valY], "full_dataset")

#   -----------------------------FULL DS DATAGEN AUGMENTED----------------------------------------------
augmented_path = "../Dataset - OCT/OCT2017/train"
augmented_data = []

class_categories = os.listdir(augmented_path)
class_categories.reverse()
print(class_categories)

create_dataset("aug",class_categories, augmented_path, augmented_data)

augX = pickle.load(open("augX.pickle", "rb"))
augY = pickle.load(open("augY.pickle", "rb"))

full_dsX = pickle.load(open("full_datasetX.pickle", "rb"))
full_dsY = pickle.load(open("full_datasetY.pickle", "rb"))

dataset_combiner([full_dsX, augX], [full_dsY, augY], "full_aug_dataset")

print('\n-+-+-+[Datasets Created & Exported]-+-+-+')
