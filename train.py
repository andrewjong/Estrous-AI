import json
import os

import torch
import torch.nn
import torch.optim

import src.utils as utils
from common_constants import EXPERIMENTS_ROOT
from metrics import create_all_metrics
from predict import create_predictions
from src.trainable import Trainable
from src.utils import build_attr, build_model

TRAIN_RESULTS_FNAME = "train.csv"


def build_and_train_model(model_starter_file=False):
    """ This is the main training function that constructs the model and trains
    it on the chosen dataset.
    """
    # Obtain train and validation datasets and dataloaders
    image_size = 299 if "inception" in args.model[0] else 224
    datasets, dataloaders = utils.get_datasets_and_loaders(
        args.data_dir, "train", "val", batch_size=args.batch_size,
        image_size=image_size)

    num_classes = len(datasets["train"].classes)

    # Instantiate the model object
    try:
        trainable = build_trainable(num_classes, args.model, args.optimizer,
                                    args.criterion, args.lr_scheduler,
                                    transfer_technique=args.transfer_technique)
        # load in existing weights if requested
        if model_starter_file:
            load_pretrained_model_weights(trainable.model, model_starter_file)
    except TypeError as e:
        print("Caught TypeError when instantiating trainable. Make sure " +
              "all required arguments are passed.\n")
        raise e

    # Make results dir path. Check if it already exists.
    try:
        os.makedirs(outdir, exist_ok=args.overwrite)
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
    trainable.train(dataloaders, args.num_epochs, 
                    results_filepath=results_filepath)
    trainable.save(outdir, extra_meta=vars(args))


def build_trainable(num_classes, model_args, optim_args, criterion_args,
                    lr_scheduler_args,
                    transfer_technique=None):
    # set the num_classes arg
    # model_args.append(f'num_classes={num_classes}')
    model = build_model(model_args, num_classes, transfer_technique)

    # do transfer learning if desired
    if transfer_technique == "finetune":
        model_params = model.parameters()
    elif transfer_technique == "fixed":
        model_params = model.fc.parameters()

    optimizer = build_attr(torch.optim, optim_args, first_arg=model_params)
    criterion = build_attr(torch.nn, criterion_args)
    lr_scheduler = build_attr(torch.optim.lr_scheduler,
                              lr_scheduler_args, first_arg=optimizer)

    trainable = Trainable(model, criterion, optimizer, lr_scheduler)
    return trainable


def load_pretrained_model_weights(target_model, load_file):
    """Load pretrained model weights except for the last fully connected layer

    Arguments:
        target_model {[type]} -- [description]
        load_file {[type]} -- [description]
    """

    pretrained_dict = torch.load(
        load_file, map_location=lambda storage, loc: storage)

    model_dict = target_model.state_dict()
    excluded = ['fc.weight', 'fc.bias']
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k not in excluded
    }
    model_dict.update(pretrained_dict)
    target_model.load_state_dict(model_dict)


def prepare_results_file(results_filepath=None):
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """
    if not results_filepath:
        results_filepath = os.path.join(outdir, TRAIN_RESULTS_FNAME)
    header = "steps,loss,train_acc,val_acc"
    results_filepath = utils.make_csv_with_header(results_filepath, header)
    return results_filepath


if __name__ == '__main__':
    from train_args import train_args
    global args
    args = train_args
    print()
    print(f'*** BEGINNING EXPERIMENT: "{args.experiment_name}" ***',
          end="\n\n")

    # load args from a previous train session if requested
    if args.load_args:
        # if the user passed in a directory, assume they meant "meta.json"
        if os.path.isdir(args.load_args):
            load_args_file = os.path.join(args.load_args, "meta.json")
        elif os.path.isfile(args.load_args):
            load_args_file = args.load_args

        with open(load_args_file, 'r') as f:
            loaded = json.load(f)
        # only load intersecting arguments, in case the load file has extraneous
        # information
        intersected_keys = set(vars(args).keys()) & set(loaded.keys())
        load_args_dict = {k: loaded[k] for k in intersected_keys}
        print("Loaded args:")
        # for each arg in argparse, check if any were not set by the user
        for common_arg, load_value in load_args_dict.items():
            # if the user did not set the argument, load it
            if not getattr(args, common_arg):
                setattr(args, common_arg, load_value)
                print(f'  {common_arg}: {load_value}')
        print()

    # get rid of the extra slash if the user put one
    if args.data_dir[-1] == "/" or args.data_dir[-1] == "\\":
        args.data_dir = args.data_dir[:-1]
    dataset_name = os.path.basename(args.data_dir)
    # Make the output directory for the experiment
    global outdir
    outdir = os.path.join(EXPERIMENTS_ROOT, args.experiment_name, dataset_name,
                          args.model[0])

    if args.use_previous_model:
        prev_model_file = os.path.join(args.load_args, "model.pth")
        print("Loading previous model from ", prev_model_file)
    else:
        prev_model_file = False

    # run the main training script
    build_and_train_model(prev_model_file)

    # calculate performance metrics with the saved model
    # print('skip_metrics: ', args.skip_metrics)
    if not args.skip_metrics:
        print()
        print("Creating predictions file...")
        predictions_file = create_predictions(outdir)
        print("Calculating performance metrics...")
        create_all_metrics(predictions_file, outdir)
    else:
        print("Skipping metrics.")

    print()
    print("Done.")
