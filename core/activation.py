# -*- coding: utf-8 -*-
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'
        

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})



