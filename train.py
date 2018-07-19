import argparse
import json
import os

import torch

import utils
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
    results_filepath = make_results_file()
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

    meta_info = {
        "experiment_name": args.experiment_name,
        "data_dir": args.data_dir,
        "model": args.model,
        "added_args": args.added_args,
        "num_epochs": args.num_epochs,
        "best_val_accuracy": best_val_acc,
        "train_accuracy": associated_train_acc,
        "train_loss": associated_train_loss
    }
    meta_out = os.path.join(outdir, META_FNAME)
    with open(meta_out, 'w') as out:
        json.dump(meta_info, out, indent=4)
    # TODO: check to see if training finished or not. if it didn't, record
    # where to pick up to finish training


def make_results_file():
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """
    results_filepath = os.path.join(outdir, TRAIN_RESULTS_FNAME)

    with open(results_filepath, 'w') as f:
        f.write("epoch,loss,train_acc,val_acc\n")
    return results_filepath


if __name__ == '__main__':
    # TODO: Convert these arguments into a configuration file later
    parser = argparse.ArgumentParser(
        description="Train an estrous cycle cell-phase classifer model.")

    parser.add_argument("model",
                        choices=list(TRAIN_MODEL_CHOICES),
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

    global args
    args = parser.parse_args()

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
