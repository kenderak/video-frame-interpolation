# -*- coding: utf-8 -*-
import sys
import os
import cv2
import keras.backend as K
import numpy as np
from scipy.misc import imresize
from networks import UNET
# Script to read video, create frames, resizing them, make predictions and make higher
# FPS (2x and 4x) videos into the 'results/' directory.

K.set_image_dim_ordering("th")
os.environ['KERAS_BACKEND'] = "tensorflow"

def load_vid(vid_path):

    cap = cv2.VideoCapture(vid_path)
    if (cap.isOpened()== False): 
        sys.exit("Error opening video!")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vid_arr = np.zeros(shape=(num_frames, 128, 384, 3), dtype="uint8")
    for i in range(num_frames):
        if i % (num_frames / 10) == 0:
            print ("Video loading is {0}% done.".format((i / (num_frames / 10) * 10)))

        ret, frame = cap.read()
        vid_arr[i] = imresize(frame, (128, 384))

    return vid_arr, fps

def double_vid_fps(vid_arr):

    # Load the model and the weights into it
    model = UNET((6, 128, 384))
    model.load_weights("./model_weights/weights_dice_loss_60ep.hdf5")

    new_vid_arr = []
    new_vid_arr.append(vid_arr[0])
    for i in range(1, len(vid_arr)):
        if i % (len(vid_arr) / 10) == 0:
            print ("FPS doubling is {0}% done.".format((i / (len(vid_arr) / 10) * 10)))

        # Predict the inner frames
        pred = model.predict(np.expand_dims(np.transpose(np.concatenate((vid_arr[i-1], vid_arr[i]), axis=2)/255., (2, 0, 1)), axis=0))
        new_vid_arr.append((np.transpose(pred[0], (1, 2, 0))*255).astype("uint8"))
        new_vid_arr.append(vid_arr[i])

    return np.asarray(new_vid_arr)

def save_vid(vid_arr, vid_out_path, fps):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_out_path, fourcc, fps, (384, 128))

    for i in range(len(vid_arr)):
        out.write(vid_arr[i])

def main():
    # Directory paths
    vid_dir = "./"
    vid_fn = "PlanetEarth_Amazing_nature_1080p.mp4"
    out_dir = "./results/"

    # Load videoframes into numpy arrays
    vid_arr, fps = load_vid(os.path.join(vid_dir, vid_fn))

    # Make the 2x FPS video by predicting inner frames with our network
    double_vid_arr = double_vid_fps(vid_arr)

    # Save videos into output directory
    save_vid(vid_arr, out_dir + vid_fn.split('.')[0] + "_resize.avi", fps=fps)
    save_vid(double_vid_arr, out_dir + vid_fn.split('.')[0] + "_double_60.avi", fps=fps*2)
    save_vid(double_vid_arr, out_dir + vid_fn.split('.')[0] + "_double_30.avi", fps=fps)

    quad_vid_arr = double_vid_fps(double_vid_arr)
    save_vid(quad_vid_arr, out_dir + vid_fn.split('.')[0] + "_quad_30.avi", fps=fps)

if __name__ == '__main__':
    main()