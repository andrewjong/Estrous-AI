import argparse
import json
import os

import torch

import src.utils as utils
from src.model_choices import TRAIN_MODEL_CHOICES

EXPERIMENTS_ROOT = "experiments"
META_FNAME = "meta.json"
TRAIN_RESULTS_FNAME = "train.csv"
MODEL_PARAMS_FNAME = "model.pth"


def main():
    # Obtain train and validation datasets and dataloaders
    datasets, dataloaders = utils.get_datasets_and_loaders(
        args.data_dir, "train", "val")
    dataset_sizes = {subset: len(datasets[subset])
                     for subset in ('train', 'val')}

    num_classes = len(datasets["train"].classes)

    # Instantiate the model object
    TrainableClass = TRAIN_MODEL_CHOICES[args.model]
    try:
        trainable = TrainableClass(num_classes, *args.added_args)
    except TypeError as e:
        print("Caught TypeError when instantiating model class. Make sure " +
              "all required model arguments are passed, in order, using the " +
              "-a flag.\n")
        print(e)
        exit(1)

    # Make results dir path. Check if it already exists.
    try:
        os.makedirs(outdir)
    except OSError:
        can_continue = input(f'The directory "{outdir}" already exists. ' +
                             "Do you wish to continue anyway? (y/N): ")
        if can_continue and can_continue.lower()[0] == "y":
            os.makedirs(outdir, exist_ok=True)
        else:
            exit()
    print(f'Writing results to "{outdir}"')

    # make the results file
    results_filepath = prepare_results_file()
    # Train
    trained_model = trainable.train(dataloaders, dataset_sizes,
                                    args.num_epochs, results_filepath)
    # Write the meta file containing train data
    write_meta(trainable.best_val_accuracy,
               trainable.associated_train_accuracy,
               trainable.associated_train_loss)

    # Save the model
    save_path = os.path.join(outdir, MODEL_PARAMS_FNAME)
    torch.save(trained_model.state_dict(), save_path)


def write_meta(best_val_acc, associated_train_acc, associated_train_loss):
    """Writes meta data about the train

    Arguments:
        best_val_acc {[type]} -- [description]
        associated_train_acc {[type]} -- [description]
        associated_train_loss {[type]} -- [description]
    """
    meta_info = vars(args)
    # add extra values to the dictionary
    meta_info.update({
        "best_val_accuracy": best_val_acc,
        "train_accuracy": associated_train_acc,
        "train_loss": associated_train_loss
    })
    meta_out = os.path.join(outdir, META_FNAME)
    with open(meta_out, 'w') as out:
        json.dump(meta_info, out, indent=4)
    # TODO: check to see if training finished or not. if it didn't, record
    # where to pick up to finish training


def prepare_results_file():
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """
    results_filepath = os.path.join(outdir, TRAIN_RESULTS_FNAME)
    header = "epoch,loss,train_acc,val_acc"
    results_filepath = utils.make_csv_with_header(results_filepath, header)
    return results_filepath


if __name__ == '__main__':
    # TODO: Convert these arguments into a configuration file later
    parser = argparse.ArgumentParser(
        description="Train an estrous cycle cell-phase classifer model.")

    parser.add_argument("model",
                        choices=list(TRAIN_MODEL_CHOICES),
                        nargs="?",
                        help=f'Choose which model to use.')
    parser.add_argument("-e", "--experiment_name",
                        default='unnamed_experiment',
                        help='Name of the experiment. This becomes the \
                        subdirectory that the experiment output is stored in, \
                        i.e. "experiments/my_experiment/" (default: \
                        "unnamed_experiment").')
    parser.add_argument("-d", "--data_dir",
                        help='Root directory of dataset to use, with classes \
                        separated into separate subdirectories.')
    parser.add_argument("-n", "--num_epochs", type=int, metavar="N",
                        default=50,
                        help='Number of epochs to train for (default: 50).')
    parser.add_argument("-a", "--added_args", nargs="+",
                        default=[],
                        help="Pass addiitional arguments for instantiating the \
                        chosen model.")
    parser.add_argument("-l", "--load_args",
                        help="Option to load in train args from a previous \
                        train session using that session's \"meta.json\" \
                        file. Passed and default args will override loaded \
                        args.")

    global args
    args = parser.parse_args()
    # load args from a previous train session if requested
    if args.load_args:
        # if the user passed in a directory, assume they meant "meta.json"
        if os.path.isdir(args.load_args):
            args_file = os.path.join(args.load_args, "meta.json")
        elif os.path.isfile(args.load_args):
            args_file = args.load_args

        with open(args_file, 'r') as f:
            loaded = json.load(f)
        intersected_keys = set(vars(args).keys()) & set(loaded.keys())
        args_dict = {k: loaded[k] for k in intersected_keys}
        print("__Loaded args:__")
        for arg, value in args_dict.items():
            if not getattr(args, arg):
                setattr(args, arg, value)
                print(f'  {arg}: {value}')

    # some setup for our eventual output directory name
    model_name = "-".join([args.model] + args.added_args)
    # get rid of the extra slash if the user put one
    if args.data_dir[-1] == "/" or args.data_dir[-1] == "\\":
        args.data_dir = args.data_dir[:-1]
    dataset_name = os.path.basename(args.data_dir)
    # Make the output directory for the experiment
    global outdir
    outdir = os.path.join(EXPERIMENTS_ROOT, args.experiment_name,
                          dataset_name, model_name)

    main()
