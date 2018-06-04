DOC = """ This script moves images into separate subdirectories (folders) for train/validation/test sets.
The images are additionally separated into folders by phase label.

This formats the data for PyTorch's torchvision.datasets.ImageFolder loader function:
https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder.

Be sure to delete existing files before rerunning the script!"
"""
"""
The folder structure will look like this: 
    train/
        proestrus/
        estrus/
        metestrus/
        diestrus/
    val/
        ... 
    test/ 
        ... 
"""
import argparse
import glob
import os
from random import Random
from shutil import copy2  # for copying files

import numpy as np
import pandas as pd
from tqdm import tqdm  # handy progress bar

cwd = os.getcwd()

# set up command line arguments
parser = argparse.ArgumentParser(description=DOC)
parser.add_argument(
    "labels_file", help="CSV file containing the labels for each image")
parser.add_argument(
    "from_dir", help="Root directory of where the unsorted images currently are")
parser.add_argument("to_dir", nargs='?', default=os.path.join("..", "data", "lavage"),
                    help="Root directory of where to put the sorted images (default: '../data/lavage/')")
parser.add_argument("-s", "--split", nargs=3, type=int, default=[60, 20, 20],
                    help="Split percentages for train, validation, and test sets respectively. E.g. '60 20 20'. \
                    Numbers must add up to 100! (default: 60 train, 20 val, 20 test).")
args = parser.parse_args()
# make sure split is valid, i.e. sums to 100
split_sum = sum(args.split)
assert split_sum == 100, "Split percentages must sum to 100. Sum=" + \
    str(split_sum) + "."

# map phase numbers to the correct phase label. this is for sorting the excel spreadsheet
PHASE_NUM_TO_LABEL = {
    "1": "proestrus",
    "2": "estrus",
    "3": "metestrus",
    "4": "diestrus"
}

# each label points to an array of filepaths to copy later
labels_to_files = {label: []
                   for label in PHASE_NUM_TO_LABEL.values()}

# read our spreadsheet
labels_df = pd.read_csv(args.labels_file, index_col=0, dtype=str)
print(f'Reading labels for {len(labels_df)} animals \
        from "{args.labels_file}".')

print(f'Reading images from "{args.from_dir}".')
print(f'Sorting files...')

# show a progress bar for file sorting
with tqdm(total=len(labels_df)) as pbar:
    # iterate over each row in the dataframe (using itertuples for speed)
    for row in labels_df.itertuples():
        animal_label = row.Index

        # iterate over each row's elements
        # (we iterate using the index because the "row" returned by itertuples doesn't store the full column name. iterating by index does)
        for i in range(1, len(labels_df.columns)):
            # get the value in the cell
            phase_num = row[i]
            # make sure the cell isn't empty
            if not pd.isnull(phase_num):
                phase_label = PHASE_NUM_TO_LABEL[str(phase_num)]
                # (columns are offset by 1 as it lacks the animal label)
                date_label = labels_df.columns[i - 1]

                # images are named starting with "AnimalNumber_Date"
                f_name = animal_label + "_" + date_label
                # includes subdirectories in search with "**"
                search_glob = os.path.join(
                    args.from_dir, "**", f_name + "*.tif")
                # match each found file to the appropriate phase label
                for file in glob.glob(search_glob, recursive=True):
                    labels_to_files[phase_label].append(file)

        pbar.update(1)

print(f'Copying files to "{args.to_dir}"...')

train_split_percent = .01 * args.split[0]
val_split_percent = .01 * args.split[1]

total_files_to_copy = sum(len(x) for x in labels_to_files.values())
# make another progress bar for copying files
with tqdm(total=total_files_to_copy) as pbar:
    for label, files in labels_to_files.items():
        # shuffle for fair training. use a seed on Random for consistency in shuffling
        Random(42).shuffle(files)
        total_files = len(files)

        # divide into train, val, test sets
        train_stop = int(train_split_percent * total_files)
        train_files = files[:train_stop]

        val_stop = int((train_split_percent + val_split_percent) * total_files)
        val_files = files[train_stop:val_stop]

        test_files = files[val_stop:]

        # make sub directories for each sub dataset and label
        for split_set in ("train", "val", "test"):
            # make the directory for the given split set
            sorted_dir = os.path.join(args.to_dir, split_set, label)
            os.makedirs(sorted_dir, exist_ok=True)
            # put the appropriate files in the directory we made
            split_set_files = locals()[split_set + "_files"]
            for f in split_set_files:
                copy2(f, sorted_dir)
                pbar.update(1)

print("Done!")
