import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # encoder --> downsampling image
    # Conv2D -> feature maps, kernel size, activation function, padding
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Double up on Convolution layers because it increases network depth (helps model learn more complex features and representations of the input data)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # decoder --> upsampling image
    # This is where you start the 'skip' steps, unique to the U-net (occurs in merge layers)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(drop5))

    # Concatenates the output of previous conv layer (up6) with output of 4th max pooling layer (drop4) along the channel axis (axis=3)
    merge6 = layers.concatenate([drop4, up6], axis=3)

    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

print('\nLoading Files...')
trainX = pickle.load(open("training_dataX.pickle", "rb"))
trainY = pickle.load(open("training_datay.pickle", "rb"))

testX = pickle.load(open("test_dataX.pickle", "rb"))
testY = pickle.load(open("test_datay.pickle", "rb"))

valX = pickle.load(open("validation_dataX.pickle", "rb"))
valY = pickle.load(open("validation_datay.pickle", "rb"))

class_dict = {0: "CLASS 1", 1: "CLASS 2", 2: "CLASS 3", 3: "CLASS 4"}

number_in_class = {"CLASS 1": 0, "CLASS 2": 0, "CLASS 3": 0, "CLASS 4": 0}

# Create the model
input_shape = (256,256,3)
model = unet(input_shape)

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50)

# Predict on the test data
y_pred = model.predict(x_test)

# Print the segmented images
for i in range(len(y_pred)):
    plt.imshow(y_pred[i].reshape(input_shape[:2]), cmap='gray')
    plt.show()