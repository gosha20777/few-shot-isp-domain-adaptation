import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, 
    Concatenate, 
    Convolution2D, 
    MaxPooling2D, 
    UpSampling2D
)


def get_model(input_shape=(None, None, 4)): 
    inputs = Input(shape=input_shape)

    conv1 = Convolution2D(16, (3, 3), activation='tanh', padding='same')(inputs)
    conv2 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Convolution2D(12, (3, 3), activation='relu', padding='same')(conv2)
    x_out = tf.nn.depth_to_space(conv3, 2)
    return Model(inputs=inputs, outputs=x_out)


def get_model_unet(input_shape=(None, None, 4)): 
    inputs = Input(shape=input_shape)

    conv1 = Convolution2D(16, (3, 3), activation='tanh', padding='same')(inputs)
    conv1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, (3, 3), activation='tanh', padding='same')(pool1)
    conv2 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv2)

    up3 = Concatenate()([Convolution2D(16, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv2)), conv1])
    conv3 = Convolution2D(12, (3, 3), activation='relu', padding='same')(up3)
    x_out = tf.nn.depth_to_space(conv3, 2)
    return Model(inputs=inputs, outputs=x_out)