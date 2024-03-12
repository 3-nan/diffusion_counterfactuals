""" Python script for converting Imagenet to the right directory structure. """
import os
import glob

data_path = "/Data/imagenet/val/"

for fpath in glob.glob(data_path + "*"):

    # Extract class
    fname, ext = os.path.splitext(fpath)
    fname = fname.split("/")[-1]
    cname = fname.split("_")[-1]

    # Make class folder
    os.makedirs(os.path.join(data_path, cname), exist_ok=True)

    # Copy and rename file
    new_fpath = os.path.join(data_path, cname, "_".join(fname.split("_")[:-1]) + ext)
    os.rename(fpath, new_fpath)
