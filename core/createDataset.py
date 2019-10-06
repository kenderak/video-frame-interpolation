# -*- coding: utf-8 -*-
from os import remove, listdir, makedirs
from os.path import exists, isdir, join
from random import shuffle
import shutil
import numpy as np
import h5py
import math
import cv2

def data_processor(row = 384, col = 128):
    data_dir         = './data'
    data_dir_resized = './data_resized'
    dataset_filename = './data/dataset.hdf5'

    resize = True
    img_new_size = (row,col)

    valid_split = 0.20
    test_split  = 0.10

    file_paths  = dict()
    num_of_samp = 0
    N = 2

    # Save train, valid and test dataset to .hdf5 file 
    def save_hdf5(sample_list, x_dataset, y_dataset, size_dataset, start_idx):
        elem_idx = start_idx
        for i in range(size_dataset):
            fin1 = open(sample_list[elem_idx][0], 'rb')
            fin2 = open(sample_list[elem_idx][1], 'rb')
            fout = open(sample_list[elem_idx][2], 'rb')
            fin1_b64 = np.fromstring(fin1.read(), np.uint8)
            fin2_b64 = np.fromstring(fin2.read(), np.uint8)
            fout_b64 = np.fromstring(fout.read(), np.uint8) 
            fin_b64  = (fin1_b64, fin2_b64)
            x_dataset[i] = fin_b64
            y_dataset[i] = fout_b64
            elem_idx += 1
        return x_dataset, y_dataset

    # Create dict of filepaths and calculate number of samples
    for folder in sorted(listdir(data_dir)):
        print("## IMG: Working directory --> " + folder + " ...")
        file_paths[folder] = []
        if not exists(join(data_dir_resized, folder)) and resize:
            makedirs(join(data_dir_resized, folder))
        for file_name in sorted(listdir(join(data_dir, folder))):
            file_paths[folder].append(file_name)

            # Load images, resize and save them
            if resize:
                img_path_load = join(data_dir, folder, file_name)
                img_path_save = join(data_dir_resized, folder, file_name)
                img_loaded  = cv2.imread(img_path_load)
                img_resized = cv2.resize(img_loaded, img_new_size)
                cv2.imwrite(img_path_save,img_resized)

        num_of_samp += len(file_paths[folder]) - N

    if not resize:
        data_dir_resized = data_dir
        
    # Read image paths into a list
    samp_idx = 0
    samp_list= []
    for key in file_paths:
        for i in range(len(file_paths[key]) - N):
            input_1_path = join(data_dir_resized, key, file_paths[key][i])
            input_2_path = join(data_dir_resized, key, file_paths[key][i+2])
            output_path  = join(data_dir_resized, key, file_paths[key][i+1])
            samp_list.append([input_1_path, input_2_path, output_path])
            samp_idx += 1

    # Shuffle samples
    shuffle(samp_list)

    # Create HDF5 file for dataset
    if exists(dataset_filename):
        print('## HDF5: Dataset deleted.')
        remove(dataset_filename)
    f = h5py.File(dataset_filename)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    train_size = int(num_of_samp*(1-(test_split+valid_split)))
    valid_size = int(num_of_samp*valid_split)+1
    test_size  = int(num_of_samp*test_split)+1

    x_train_dataset = f.create_dataset('x_train', (train_size,2,), dtype=dt)
    y_train_dataset = f.create_dataset('y_train', (train_size,), dtype=dt)
    x_valid_dataset = f.create_dataset('x_valid', (valid_size,2,), dtype=dt)
    y_valid_dataset = f.create_dataset('y_valid', (valid_size,), dtype=dt)
    x_test_dataset  = f.create_dataset('x_test' , (test_size,2,), dtype=dt)
    y_test_dataset  = f.create_dataset('y_test' , (test_size,), dtype=dt)

    # Save data to hdf5
    print("## HDF5: Saving TRAIN data ...")
    x_train_dataset, y_train_dataset = save_hdf5(samp_list, x_train_dataset, y_train_dataset, train_size, start_idx=0)
    print("## HDF5: Saving VALIDATION data ...")
    x_valid_dataset, y_valid_dataset = save_hdf5(samp_list, x_valid_dataset, y_valid_dataset, valid_size, start_idx=train_size+1)
    print("## HDF5: Saving TEST data ...")
    x_test_dataset,  y_test_dataset  = save_hdf5(samp_list, x_test_dataset, y_test_dataset, test_size, start_idx=num_of_samp-test_size)

    # Remove resized directory
    shutil.rmtree(data_dir_resized)
