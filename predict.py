import argparse
import json
import os

import torch
from tqdm import tqdm

import src.utils as utils
from src.model_choices import TRAIN_MODEL_CHOICES
from train import EXPERIMENTS_ROOT
from train import META_FNAME
from train import MODEL_PARAMS_FNAME

PREDICT_FNAME = "predictions.csv"


def main():
    # load in the meta data created by training
    meta_path = os.path.join(args.load_dir, META_FNAME)
    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)

    # if the user specifies a different dataset to predict on for some reason,
    # maybe for fun, use it. else just use the dataset specified in the meta
    data_dir = args.data_dir if args.data_dir else meta_dict["data_dir"]
    # Get our DataLoader
    datasets, dataloaders = utils.get_datasets_and_loaders(
        data_dir, args.subset, include_paths=True)
    dataset = datasets[args.subset]
    num_classes = len(dataset.classes)
    dataloader = dataloaders[args.subset]

    # load the model using the meta data
    try:
        model = load_model(meta_dict, num_classes)
    except FileNotFoundError:
        print("No model file found. Did training finish? " +
              f'Make sure the model file is named "{MODEL_PARAMS_FNAME}".')
        exit(1)

    # move to device and set to eval mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # put the image name, the classes, the predicted class, and the label
    header = ','.join(['image_name'] + dataset.classes +
                      ['predicted', 'label'])
    results_file = make_results_file(header)
    print("Writing results to", results_file)

    with tqdm(desc="Predict", total=len(dataset)) as pbar:
        # for all the inputs, make a prediction
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                # make prediction by passing forward through the model
                raw_outputs = model(inputs)
                outputs = torch.nn.Softmax(dim=1)(raw_outputs)
                _, predictions = torch.max(outputs, 1)
                for i, row in enumerate(outputs):
                    # keep only the first 4 decimals
                    values_as_strings = [f'{value:.4f}'
                                         for value in row.numpy()]
                    image_name = os.path.basename(paths[i])
                    # get the actual class names instead of just indices
                    predicted_class_name = dataset.classes[predictions[i]]
                    label_class_name = dataset.classes[labels[i]]
                    # add the class names to the array
                    values_as_strings.insert(0, image_name)
                    values_as_strings.extend(
                        [predicted_class_name, label_class_name])
                    # make the array a single csv string
                    csv_line = ','.join(values_as_strings)
                    # write the file
                    with open(results_file, 'a') as f:
                        f.write(csv_line + '\n')

            pbar.update(inputs.size(0))


def load_model(meta_dict, num_classes):
    """Load a model using the meta file created during training.
    First uses the model name and added arguments to construct the
    architecture. Then loads in the parameter weights using the model file.

    Arguments:
        meta_dict {dict} -- a dictionary constructed from the meta.json file
        num_classes {int} -- [description]

    Returns:
        [type] -- [description]
    """

    HyperParamsClass = TRAIN_MODEL_CHOICES[meta_dict["model"]]
    model = HyperParamsClass(num_classes, *meta_dict["added_args"]).model

    model_params = os.path.join(args.load_dir, MODEL_PARAMS_FNAME)
    model.load_state_dict(torch.load(model_params))
    return model


def make_results_file(header):
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """
    results_filepath = os.path.join(args.load_dir, PREDICT_FNAME)
    # write the csv header
    with open(results_filepath, 'w') as f:
        f.write(header + "\n")
    return results_filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="See model predictions and class probabilities on a \
        data subset.")
    parser.add_argument("load_dir",
                        help="Directory to load the model from. Typically a " +
                        f'leaf directory under "{EXPERIMENTS_ROOT}/""')
    parser.add_argument("-d", "--data_dir",
                        default=False,
                        help="Root directory of dataset to use, with classes \
                        separated into separate subdirectories. Default is \
                        the dataset used for training.")
    parser.add_argument("-s", "--subset", default="val",
                        help="Which subset of the dataset to evaluate on. " +
                        "E.g. 'train', 'val', or 'test' (default: 'val').")
    global args
    args = parser.parse_args()

    main()
