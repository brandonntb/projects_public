import glob
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tqdm import tqdm

import random


def aug_paths(class_list, increase):
    temp = []

    while len(temp) < increase:
        if len(temp) >= len(class_list) or len(class_list) > increase:
            temp.append(random.choice(class_list))
        else:
            for i in range(len(class_list)):
                temp.append(class_list[i])

    return temp

def augmentor(image_list, save_path):
    count = 0
    for path in tqdm(image_list):
        img = load_img(path, color_mode='grayscale')
        img_array = img_to_array(img)
        reshaped = img_array.reshape((1,) + img_array.shape)

        for i in generator.flow(reshaped, batch_size=1, save_to_dir=save_path,
                                  save_prefix="DME_" + str(count), save_format="jpeg"):
            count += 1

            break

        if count > len(image_list):
            break

# generator = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
#                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
#                                vertical_flip=True, fill_mode='nearest', brightness_range=[0.6, 1.4])

generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                               shear_range=0.2, zoom_range=0.1, horizontal_flip=True,
                               vertical_flip=True, fill_mode='nearest', brightness_range=[0.4, 1.4])


dme_paths = glob.glob("../Dataset - OCT/OCT2017/train/DME/*.jpeg")
drusen_paths = glob.glob("../Dataset - OCT/OCT2017/train/DRUSEN/*.jpeg")
normal_paths = glob.glob("../Dataset - OCT/OCT2017/train/NORMAL/*.jpeg")
cnv_paths = glob.glob("../Dataset - OCT/OCT2017/train/CNV/*.jpeg")

paths = [dme_paths, drusen_paths, normal_paths, cnv_paths]

class_sizes = [len(i) for i in paths]

print(class_sizes)

minority_classes = [i for i in paths if len(i) != max(class_sizes)]

target_size = max(class_sizes)

print(target_size)

increases = [target_size - i for i in class_sizes if i != target_size]


drusen_save_path = "../Dataset - OCT/Augmented_datasets/DRUSEN"
dme_save_path = "../Dataset - OCT/Augmented_datasets/DME"
normal_save_path = "../Dataset - OCT/Augmented_datasets/Normal"


dme_aug = aug_paths(minority_classes[0], increases[0])
drusen_aug = aug_paths(minority_classes[1], increases[1])
normal_aug = aug_paths(minority_classes[2], increases[2])


augmentor(dme_aug, dme_save_path)
augmentor(drusen_aug, drusen_save_path)
augmentor(normal_aug, normal_save_path)



