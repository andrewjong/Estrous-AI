"""This file can run as a standalone script to create a model predictions file 
by running inputs through a model. This file can also be imported as a module. 
The main function is "create_predictions".
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

import src.utils as utils
from common_constants import EXPERIMENTS_ROOT, META_FNAME, MODEL_PARAMS_FNAME
from src.model_choices import TRAIN_MODEL_CHOICES

# name for output. will have subset prepended to it later
PREDICT_BASENAME = "predictions.csv"

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_predictions(load_dir, subset='val', alternate_data_dir=False):
    """Create predictions file using the model from an experiment directory.
    Uses the experiment directory's "meta.json" file to load model
    architecture and path to dataset.

    Arguments:
        load_dir {string} -- experiment directory path that contains experiment
        files
        subset {string} -- either 'train', 'val', or 'test', i.e. which subset
        to run predictions on

    Keyword Arguments:
        alternate_data_dir {bool} -- whether to run the model on a different
        dataset than is specified in 'meta.json' (default: {False})
    """
    # load in the meta data created by training
    meta_dict = get_meta_dict_from_load_dir(load_dir)
    # if the user specifies a different dataset to predict on for some reason,
    # maybe for fun, use it. else just use the dataset specified in the meta
    data_dir = alternate_data_dir if alternate_data_dir \
        else meta_dict["data_dir"]

    dataset, dataloader = get_subset_dataset_and_loader(data_dir, subset)

    # load the model using the meta data
    model_path = os.path.join(load_dir, MODEL_PARAMS_FNAME)
    num_classes = len(dataset.classes)
    device = get_device()
    model = load_model(model_path, meta_dict, num_classes, device)

    # get where we will write results to
    results_file = prepare_results_csv(load_dir, subset, dataset.classes,
                                       alternate_data_dir)
    print("Writing results to", results_file)

    with tqdm(desc="Predict", total=len(dataset)) as pbar:
        # for all the inputs, make a prediction
        for inputs, labels, input_paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # make prediction by passing forward through the model
            with torch.set_grad_enabled(False):
                raw_outputs = model(inputs)
                outputs = torch.nn.Softmax(dim=1)(raw_outputs)

            write_batch_prediction(dataset.classes, input_paths, labels,
                                   outputs, results_file)

            pbar.update(inputs.size(0))
    # return the path to the results file
    return results_file


def get_meta_dict_from_load_dir(load_dir):
    """Loads the meta dictionary from the experiment's load directory, 
    specifically from the meta.json file.
    
    Arguments:
        load_dir {string} -- leaf directory containing experiment files
    
    Returns:
        dict -- dictionary parsed from meta.json file
    """
    meta_path = os.path.join(load_dir, META_FNAME)
    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)
    return meta_dict


def get_subset_dataset_and_loader(data_dir, subset, batch_size=4):
    """Returns the single dataset and dataloader for the specfied subset

    Arguments:
        subset {string} -- either "train" or "val" or "test"
        data_dir {string} -- top level directory to load data from

    Returns:
        torch.Dataset, torch.DataLoader -- the dataset and dataloader for that 
        subset
    """

    # Get our DataLoader
    datasets, dataloaders = utils.get_datasets_and_loaders(
        data_dir, subset, include_paths=True, batch_size=batch_size)
    dataset = datasets[subset]
    dataloader = dataloaders[subset]

    return dataset, dataloader


def load_model(model_path, meta_dict, num_classes, device="cpu"):
    """Load a model using the meta file created during training.
    First uses the model name and added arguments to construct the
    architecture. Then loads in the parameter weights using the model file.
    Sets the model in eval mode, on GPU if present, on CPU if not.

    Arguments:
        meta_dict {dict} -- a dictionary constructed from the meta.json file
        num_classes {int} -- number of classes the model outputs

    Returns:
        torch.NN -- the torch model object
    """
    HyperParamsClass = TRAIN_MODEL_CHOICES[meta_dict["model"]]
    model = HyperParamsClass(num_classes, *meta_dict["added_args"]).model
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=lambda storage, loc: storage))
    except FileNotFoundError:
        print("No model file found. Did training finish? " +
              f'Make sure the model file is named "{MODEL_PARAMS_FNAME}".')
        exit(1)

    # move to GPU or CPU and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def prepare_results_csv(load_dir,
                        subset,
                        class_names,
                        alternate_data_dir=False):
    """Prepares the results csv by creating the file and adding a column header.

    If alternate_data_dir is used, the name of that dataset is prepended to the
    file name.

    Arguments:
        load_dir {string} -- leaf directory containing experiment files
        data_dir {string} -- top level directory to load data from
        subset {string} -- either "train" or "val" or "test"
        class_names {iterable} -- list of class names

    Keyword Arguments:
        alternate_data_dir {bool} -- different directory (default: {False})

    Returns:
        string -- path to the created file
    """
    # create the file path
    name_parts = [subset, PREDICT_BASENAME]
    # if a different dataset was specified, include it in the filename to
    # differentiate it
    if alternate_data_dir:
        name_parts.insert(0, os.path.basename(alternate_data_dir))

    out_name = "_".join(name_parts)
    # output in the experiment directory
    results_filepath = os.path.join(load_dir, out_name)

    # put the image name, the classes, the predicted class, and the label
    header = ','.join(['image_name'] + class_names + ['label', 'predicted'])
    utils.make_csv_with_header(results_filepath, header)

    return results_filepath


def write_batch_prediction(class_names, input_paths, labels, outputs,
                           results_file):
    """Write labels, and outputs to file
    
    Arguments:
        input_paths {iterable} -- file paths of the inputs
        labels {torch.Tensor} -- 1 dimensional tensor of the correct labels
        outputs {torch.Tensor} -- tensor holding model outputs
    """
    _, predictions = torch.max(outputs, 1)

    # iterate through each row in the batch
    for index, row in enumerate(outputs):
        # only get the file basename, not the whole path
        image_name = os.path.basename(input_paths[index])
        # get the actual string names of the class
        label_class = class_names[labels[index]]
        predicted_class = class_names[predictions[index]]

        write_row_prediction(image_name, row, label_class, predicted_class,
                             results_file)


def write_row_prediction(image_name, row, label_class_name,
                         predicted_class_name, results_file):
    """Writes a single prediction row to file in csv format.
    Written format:
    "image_name, [row_values], label_class, predicted_class"
    
    Arguments:
        image_name {string} -- name of the file
        row_values {torch.Tensor} -- one dimensional tensor containing outputs
                                for each class
        label_class {string} -- the correct labeled class name
        predicted_class {string} -- the class name of the prediction
    """
    # keep only the first 4 decimals
    row_values_as_strings = [f'{value:.4f}' for value in row.cpu().numpy()]
    # get the actual class names instead of just indices
    # add the class names to the array
    row_values_as_strings.insert(0, image_name)
    row_values_as_strings.extend([label_class_name, predicted_class_name])
    # make the array a single csv string
    csv_line = ','.join(row_values_as_strings)
    # write the file
    with open(results_file, 'a') as f:
        f.write(csv_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="See model predictions and class probabilities on a \
        data subset.")
    parser.add_argument(
        "load_dir",
        help="Directory to load the model from. Typically a " +
        f'leaf directory under "{EXPERIMENTS_ROOT}/""')
    parser.add_argument(
        "-d", "--data_dir", default=False,
        help="Root directory of dataset to use, with classes \
                        separated into separate subdirectories. Default is \
                        the dataset used for training.")
    parser.add_argument(
        "-s",
        "--subset",
        default="val",
        help="Which subset of the dataset to evaluate on. " +
        "E.g. 'train', 'val', or 'test' (default: 'val').")
    args = parser.parse_args()

    create_predictions(args.load_dir, args.subset, args.data_dir)
