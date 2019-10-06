# -*- coding: utf-8 -*-
import os.path
import h5py
import json
import pickle

import keras.backend as K
from keras.optimizers import SGD, adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
K.set_image_dim_ordering("th")

from core.createDataset import data_processor
from core.imageGenerator import train_generator, valid_generator
from core.networks import UNET
from core.losses import charbonnier, soft_dice


def get_hdf5(filename):
    # Reading dataset
    hdf5_file = './data/' + filename

    if not os.path.isfile(hdf5_file):
        data_processor()
    return h5py.File(hdf5_file, 'r')

def save_history(obj, name):
    try:
        filename = open(name + ".pickle","wb")
        pickle.dump(obj, filename)
        filename.close()
        return(True)
    except:
        return(False)

def main():

    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    hf = get_hdf5(configs['data']['filename'])

    x_train = hf.get('x_train')
    y_train = hf.get('y_train')
    x_valid = hf.get('x_valid')
    y_valid = hf.get('y_valid')

    nb_epochs = configs['training']['epochs']
    batch_size= configs['training']['batch_size']

    # Currenty U-net is the only implemented network. By default the size of images is 384x128.
    model = UNET((6,128,384), configs['layer']['activation'])

    # We use ADAM optimizer and custom loss functions as 'charbonnier' and 'soft_dice'
    optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss_f    = configs['model']['loss']
    if loss_f == "soft_dice":
        loss = soft_dice
    if loss_f == "charbonnier":
        loss = charbonnier

    # Compile the model. We use accuracy also but it is not so good due to the high number of classes.
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    # We only save the best model and we reduce learning rate when the val_loss is not getting
    # better under 10 epoch. It is for SGD.
    callbacks = [
            ModelCheckpoint(filepath="./" + configs['model']['save_dir'] + "/" + configs['model']['file_name'] + ".hdf5", monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
    ]

    # Train the model
    hist = model.fit_generator(
        generator=train_generator(x_train,y_train, batch_size),
        steps_per_epoch = x_train.shape[0]/batch_size,
        validation_data=valid_generator(x_valid, y_valid, batch_size),
        validation_steps= x_valid.shape[0]/batch_size,
        epochs = 1,
        callbacks=callbacks
    )

    ## Save the history
    save_history(hist, "./" + configs['model']['save_dir'] + "/" +configs['model']['file_name'])

if __name__ == '__main__':
    main()