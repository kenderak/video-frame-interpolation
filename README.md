# Deep Learning with Python
### Project: Video Frame Interpolation with Deep Convolutional Neural Network

The following scripts were created for the Deep Learning 2018 course from BUTE by József Kenderák, Árom Görög and Dániel Határ.

## Prerequisites
Anaconda 5.3 contains a lot of libraries but we need the followings:
```
conda install -c conda-forge keras
conda install -c conda-forge tensorflow
conda install -c conda-forge opencv
conda install -c anaconda scipy
```

## Directory structure and files
```
core/createDataset.py  - Put the whole dataset into the HDF5 file structuring by train, valid and test sets
core/imageGenerator.py - Train and Validation generators for *.fit_generator()
core/losses.py         - Defined some new Loss functions
core/networks.py       - Currently only contains U-net architecture
core/activation.py     - Defined custom activation layer SWISH
data/                  - The whole RAW dataset
model_weights/         - Wieghts of the best model
results/               - Save folder of the videos after testing
train.py               - Train the model after preprocessing
test.py                - After training we can test the network with videos
config.json            - Contain configuration of the model and training parameters
```

## HowTo
### 1. Step - Preprocessing
Copy the previously shared dataset to the root directory and just run:
```
data_processor(row,col) // now this function will do it all, you do not have to run createDataset.py directly
```
In the *.py* file you can change the resize resolution of the images by *img_new_size = (384,128)*. The image resolution by default is 384x128, it is important due to input size of the neural network. After the preprocess is done, you can see the *dataset.hdf5* file in the root directory.

### 2. Step - Training
After the preprocessing is done just run:
```
python train.py
```
You must nothing to change in *train.py*! Change *config.json* to modify your training parameters!

### 3. Step - Testing
After the training you can test your model via videos. Just download an *.mp4* video to the root directory and in the *test.py* you should add the filename in the *main()* function to the *vid_fn* variable. Make sure you add your *.hdf5* file path of your model correctly in the *.load_weights(...)* line.
```
python test.py
```
After testing is done, you can look at the predicted videos in the *results/* directory.

## TODO
 - [x] Train on batch
 - [x] Make the usage more comfortable
 - [x] Make better dir and file structure
 - [x] Save the history into a file after training
 - [x] Implement new activation layer
 - [ ] Implement custom pooling layer
