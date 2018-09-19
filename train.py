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
from src.utils import build_attr, build_model_from_args


def build_and_train_model(model_starter_file=False):
    """ This is the main training function that constructs the model and trains
    it on the chosen dataset.
    """
    # Instantiate the trainable object
    try:
        trainable = build_trainable_from_args(
            args.data_dir,
            args.model,
            args.optimizer,
            args.criterion,
            args.lr_scheduler,
            args.batch_size,
            outdir,
        )
        # load in existing weights if requested
        if model_starter_file:
            trainable.load_model_weights(model_starter_file)
    except TypeError as e:
        print(
            "Caught TypeError when instantiating trainable. Make sure "
            + "all required arguments are passed.\n"
        )
        raise e

    # Make results dir path. Check if it already exists.
    try:
        os.makedirs(outdir, exist_ok=args.overwrite)
    except OSError:
        can_continue = input(
            f'The directory "{outdir}" already exists. '
            + "Do you wish to continue anyway? (y/N): "
        )
        if can_continue and can_continue.lower()[0] == "y":
            os.makedirs(outdir, exist_ok=True)
        else:
            exit()
    print(f'Writing results to "{outdir}"')

    # Train
    trainable.train(args.num_epochs)
    trainable.save(extra_meta=vars(args))
    return trainable


def build_trainable_from_args(
    datadir,
    model_args,
    optim_args,
    criterion_args,
    lr_scheduler_args,
    batch_size,
    outdir,
):
    """Builds a Trainable from command line args

    Arguments:
        datadir {[type]} -- [description]
        model_args {[type]} -- [description]
        optim_args {[type]} -- [description]
        criterion_args {[type]} -- [description]
        lr_scheduler_args {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    # set the num_classes arg
    # model_args.append(f'num_classes={num_classes}')
    model_name, _ = utils.args_to_name_and_kwargs(model_args)
    image_size = utils.determine_image_size(model_name)
    dataloaders = utils.get_train_val_dataloaders(
        datadir, 0.15, image_size, args.batch_size
    )

    model = build_model_from_args(args.model)
    model = utils.fit_model_last_to_dataset(model, dataloaders["train"].dataset)

    optimizer = build_attr(torch.optim, optim_args, first_arg=model.parameters())
    criterion = build_attr(torch.nn, criterion_args)
    lr_scheduler = build_attr(
        torch.optim.lr_scheduler, lr_scheduler_args, first_arg=optimizer
    )

    trainable = Trainable(
        dataloaders, model, criterion, optimizer, lr_scheduler, outdir
    )
    return trainable


def load_args(args):
    # if the user passed in a directory, assume they meant "meta.json"
    if os.path.isdir(args.load_args):
        load_args_file = os.path.join(args.load_args, "meta.json")
    elif os.path.isfile(args.load_args):
        load_args_file = args.load_args

    with open(load_args_file, "r") as f:
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
            print(f"  {common_arg}: {load_value}")
    print()


if __name__ == "__main__":
    from train_args import train_args

    global args
    args = train_args
    print()
    print(f'*** BEGINNING EXPERIMENT: "{args.experiment_name}" ***', end="\n\n")

    # load args from a previous train session if requested
    if args.load_args:
        load_args(args)

    # get rid of the extra slash if the user put one
    if args.data_dir[-1] == "/" or args.data_dir[-1] == "\\":
        args.data_dir = args.data_dir[:-1]
    dataset_name = os.path.basename(args.data_dir)
    # Make the output directory for the experiment
    global outdir
    outdir = os.path.join(
        EXPERIMENTS_ROOT, args.experiment_name, dataset_name, args.model[0]
    )

    if args.use_previous_model:
        prev_model_file = os.path.join(args.load_args, "model.pth")
        print("Loading previous model from ", prev_model_file)
    else:
        prev_model_file = False

    # run the main training script
    trainable = build_and_train_model(prev_model_file)

    # calculate performance metrics with the saved model
    # print('skip_metrics: ', args.skip_metrics)
    if not args.skip_metrics:
        print()
        print("Creating predictions file...")
        predictions_file = create_predictions(
            outdir, subset="test", data_dir=args.data_dir, model=trainable.model
        )
        print("Calculating performance metrics...")
        create_all_metrics(predictions_file, outdir)
    else:
        print("Skipping metrics.")

    print()
    print("Done.")
