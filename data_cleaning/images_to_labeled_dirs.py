""" This script moves images into separate sub-directories (folders) according to which class the image belongs to. 

This formats the data for PyTorch's torchvision.datasets.ImageFolder loader function:
https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder

The folder structure will look like this: 
    root/proestrus/C##_M_DD_image#.png
    root/proestrus/C##_M_DD_image#.png
    root/proestrus/C##_M_DD_image#.png

    root/estrus/C##_M_DD_image#.png
    root/estrus/C##_M_DD_image#.png
    root/estrus/C##_M_DD_image#.png

The four classes are: proestrus (1), estrus (2) metestrus (3), diestrus (4)

"""
import argparse
import numpy as np
import pandas as pd
import glob
import os
from shutil import copy2  # for copying files
from tqdm import tqdm  # handy progress bar

cwd = os.getcwd()

# set up command line arguments
parser = argparse.ArgumentParser(
    description="Sort estrous cycle image data to class subdirectories.")
parser.add_argument(
    "labels_file", help="Excel file containing the labels for each image")
parser.add_argument(
    "from_dir", help="Root directory of where images currently are")
parser.add_argument("--to_dir", default=os.path.join("..", "data", "lavage_images"),
                    help="Root directory of where to put the sorted images (default: '../data/lavage_images/')")
args = parser.parse_args()


# map phase numbers to the correct phase label
PHASE_NUM_TO_LABEL = {
    "1": "proestrus",
    "2": "estrus",
    "3": "metestrus",
    "4": "diestrus"
}

# make sub directories for each label
for _, label in PHASE_NUM_TO_LABEL.items():
    label_dir = os.path.join(args.to_dir, label)
    os.makedirs(label_dir, exist_ok=True)

# read our spreadsheet
labels_df = pd.read_csv(args.labels_file, index_col=0, dtype=str)
print(f'Reading labels for {len(labels_df)} animals from "{args.labels_file}".')

print(f'Taking images from "{args.from_dir}".')
print(f'Sorting and copying files to "{args.to_dir}"...')
# show a progress bar
with tqdm(total=len(labels_df)) as pbar:
    # iterate over each row in the dataframe (using tuples for speed)
    for row in labels_df.itertuples():
        animal_label = row.Index

        # iterate over each row's element
        # (we iterate like this because the "row" returned by itertuples doesn't store the full column name.
        # this way includes it)
        for i in range(1, len(labels_df.columns)):
            phase_num = row[i]
            if not pd.isnull(phase_num):
                phase_label = PHASE_NUM_TO_LABEL[str(phase_num)]
                # columns are offset by 1 due to no animal label
                date_label = labels_df.columns[i - 1]

                f_name = animal_label + "_" + date_label
                # glob includes subdirectories
                search_glob = os.path.join(
                    args.from_dir, "**", f_name + "*.tif")

                for file in glob.glob(search_glob, recursive=True):
                    labeled_dir = os.path.join(args.to_dir, phase_label)
                    copy2(file, labeled_dir)

        pbar.update(1)
print("Done!")
