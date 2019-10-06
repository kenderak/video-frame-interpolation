# -*- coding: utf-8 -*-
# Model architectures
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate, Activation
from keras.models import Model
from core.activation import Swish

def UNET(input_shape, activation = 'relu'):
    '''
    Modified U-net architecture with some extra blocks (+Conv and BatchNorm layer)

    # Arguments
    input shape: Input size (a tuple) of the network. Format: (2*channels, height, width)
    because it will get 2 images.

    # Output
    model: The created U-net network with output_size = (channels, height, width)
    '''

    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation=activation, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation=activation, padding='same')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    conv2 = Conv2D(64, (3, 3), activation=activation, padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation=activation, padding='same')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    conv3 = Conv2D(128, (3, 3), activation=activation, padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation=activation, padding='same')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    conv4 = Conv2D(256, (3, 3), activation=activation, padding='same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation=activation, padding='same')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
    conv5 = Conv2D(512, (3, 3), activation=activation, padding='same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation=activation, padding='same')(bn5)
    bn5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)
    conv5_2 = Conv2D(512, (3, 3), activation=activation, padding='same')(pool5)
    bn5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Conv2D(512, (3, 3), activation=activation, padding='same')(bn5_2)
    bn5_2 = BatchNormalization()(conv5_2)
    up5_2 = concatenate([UpSampling2D(size=(2, 2))(bn5_2), bn5], axis=1)
    conv6_2 = Conv2D(512, (3, 3), activation=activation, padding='same')(up5_2)
    bn6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Conv2D(512, (3, 3), activation=activation, padding='same')(bn6_2)
    bn6_2 = BatchNormalization()(conv6_2)
    up6 = concatenate([UpSampling2D(size=(2, 2))(bn6_2), bn4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation=activation, padding='same')(up6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation=activation, padding='same')(bn6)
    bn6 = BatchNormalization()(conv6)
    up7 = concatenate([UpSampling2D(size=(2, 2))(bn6), bn3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation=activation, padding='same')(up7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation=activation, padding='same')(bn7)
    bn7 = BatchNormalization()(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(bn7), bn2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation=activation, padding='same')(up8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation=activation, padding='same')(bn8)
    bn8 = BatchNormalization()(conv8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(bn8), bn1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation=activation, padding='same')(up9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation=activation, padding='same')(bn9)
    bn9 = BatchNormalization()(conv9)
    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(bn9)

    model = Model(inputs=inputs, outputs=conv10)

    return model