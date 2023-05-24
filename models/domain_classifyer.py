from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense, BatchNormalization
from models.reverce_gradient import GradientReversal


def get_model(input_shape=(None, None, 512), hp_lambda=1.0): 
    inputs = Input(shape=input_shape)
    grad_reverse = GradientReversal(hp_lambda=hp_lambda)(inputs)

    conv = Convolution2D(512, (3, 3), activation='relu', padding='same')(grad_reverse)
    conv = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv)
    pool = GlobalAveragePooling2D()(conv)
    dense1 = Dense(256, activation='relu')(pool)
    bn1 = BatchNormalization()(dense1)
    outputs = Dense(1, activation='sigmoid')(bn1)

    return Model(inputs=inputs, outputs=outputs)