"""
Example of how predicted images can be converted into an H5 file. Notice that,
connected components is applied to label different mitochondria to match instance 
segmentation problem.

You should modify the following variables:
    - pred_dir : path to the directory from which the images will be read
    - h5file_name : name of the H5 file to be created (follow the instructions in
                    https://mitoem.grand-challenge.org/Evaluation/ to name the 
                    files accordingly)
The H5 file should be saved in the directory where this script was called
"""

import os                                                                       
import h5py
import numpy as np                                                              
from skimage.io import imread
from tqdm import tqdm
from skimage import measure, feature
from scipy import ndimage
from PIL import ImageEnhance, Image
from os import path

img_shape = (4096, 4096)
img_out_shape = (2048, 2048)
pred_dir = 'binarized_50ov'
pred_ids = sorted(next(os.walk(pred_dir))[2])  
h5file_name = '0_human_instance_seg_pred.h5'

# Allocate memory for the predictions
pred_stack = np.zeros(img_out_shape + (len(pred_ids),), dtype=np.int64)

# Read all the predictions
print("Creating {}".format(h5file_name))                                        
for n, id_ in tqdm(enumerate(pred_ids)):
    img = imread(os.path.join(pred_dir, id_))

    # Resize the image 
    img = img.astype('uint8')
    img = Image.fromarray(img)
    img = img.resize((2048,2048))                                           
    img = np.array(img)

    pred_stack[n] = img

# Apply connected components to make instance segmentation
pred_stack = (pred_stack / 255).astype('int64')
pred_stack, nr_objects = ndimage.label(pred_stack)
print("Number of objects is {}".format(nr_objects))

# Create the h5 file (using lzf compression to save space)
h5f = h5py.File(h5file_name, 'w')
h5f.create_dataset('dataset_1', data=pred_stack, compression="lzf")
h5f.close()

