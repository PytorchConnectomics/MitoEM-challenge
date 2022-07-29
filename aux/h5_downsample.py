# Example:
# python -u h5_downsample.py -i 0_human_instance_seg_pred.h5 -o 0_human_instance_seg_pred_v2.h5

import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input h5 file")
parser.add_argument("-o", "--output", required=True, help="Output h5 downsample file")
parser.add_argument("-d", "--down", help="[z,y,x] downsampling to be applied", default=[1,2,2])
args = parser.parse_args()

# Load H5 file
h5f = h5py.File(args.input, 'r')
k = list(h5f.keys())

# Change the key accordingly
print("Keys: {}".format(k))
data = np.array(h5f[k[0]])
print("Loaded data shape: {}".format(data.shape))

data = data[::args.down[0],::args.down[1],::args.down[2]]
print("Downsampled data shape: {}".format(data.shape))

out_dir = os.path.dirname(args.output)
os.makedirs(out_dir, exist_ok=True)

# Create the h5 file (using lzf compression to save space)
h5f = h5py.File(args.output, 'w')
h5f.create_dataset('main', data=data, compression="lzf")
h5f.close()
print("File saved in {}".format(args.output))
