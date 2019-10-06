# -*- coding: utf-8 -*-
# This file contain some custom loss functions
from keras import backend as K

_EPSILON = K.epsilon()

def charbonnier(y_true, y_pred):
    ''' 
    Charbonnier loss function for arbitrary batch size, and number of spatial dimensions.
    Assumes the `channels_first` format.

    # Arguments
        y_true: Ground truth image
        y_pred: Predicted image, output of the network
    '''

    return K.sqrt(K.square(y_true - y_pred) + 0.01**2)

def soft_dice(y_true, y_pred):
    ''' 
    Soft dice loss calculation for arbitrary batch size, and number of spatial dimensions.
    Assumes the `channels_first` format. _EPSILON is used for numerical stability to avoid
    divide by zero errors.

    # Arguments
        y_true: Ground truth image
        y_pred: Predicted image, output of the network
    '''

    axes = tuple(range(1, len(y_pred.shape))) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean(numerator / (denominator + _EPSILON))