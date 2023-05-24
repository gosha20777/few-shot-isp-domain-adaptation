from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Convolution2D


def get_model(input_shape=(None, None, 4)): 
    inputs = Input(shape=input_shape)

    conv1 = Convolution2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv2)

    return Model(inputs=inputs, outputs=conv3)