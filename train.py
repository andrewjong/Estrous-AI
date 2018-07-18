import argparse
import copy
import os
import time

import torch
from tqdm import tqdm

import utils
from src.model_choices import TRAIN_MODEL_CHOICES

EXPERIMENTS_ROOT = "experiments"
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

args = parser.parse_args()

# Make the output directory for the experiment
model_name = "-".join([args.model] + args.added_args)
# get rid of the extra slash if the user put one
if args.data_dir[-1] == "/" or args.data_dir[-1] == "\\":
    args.data_dir = args.data_dir[:-1]
dataset_name = os.path.basename(args.data_dir)

outdir = os.path.join(EXPERIMENTS_ROOT, args.experiment_name,
                      dataset_name, model_name)


def main():
    # obtain train and validation datasets and dataloaders
    datasets, dataloaders = utils.get_datasets_and_loaders(
        args.data_dir, "train", "val")
    dataset_sizes = {subset: len(datasets[subset])
                     for subset in ('train', 'val')}

    num_classes = len(datasets["train"].classes)

    # instantiate the model object
    HyperParamsClass = TRAIN_MODEL_CHOICES[args.model]
    hparams = HyperParamsClass(num_classes, *args.added_args)

    # make results dir path
    try:
        os.makedirs(outdir)
    except OSError:
        can_continue = input(f'The directory "{outdir}" already exists. ' +
                             "Do you wish to continue anyway? (y/N): ")
        if can_continue and can_continue.lower()[0] == "y":
            os.makedirs(outdir, exist_ok=True)
        else:
            quit()
    print(f'Writing results to "{outdir}"')

    # train
    trained_model = train_model(
        hparams.model, hparams.criterion, hparams.optimizer,
        hparams.lr_scheduler, args.num_epochs,
        dataloaders, dataset_sizes)

    # Save the model
    save_path = os.path.join(outdir, "model.pth")
    torch.save(trained_model.state_dict(), save_path)


def train_model(model, criterion, optimizer, scheduler, num_epochs,
                dataloaders, dataset_sizes):
    """Train the given model.

    Arguments:
        model {[type]} -- the model to train
        criterion {[type]} -- a criterion (loss function) for the optimizer
        optimizer {[type]} -- the optimizer to use such as SGD, Adam, etc
        scheduler {[type]} -- update learning rate based on completed epochs

    Keyword Arguments:
        num_epochs {int} -- number of epochs to train for (default: {50})
    """
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # create the results file header
    results_filepath = make_results_file()

    # setup our best results to return later
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    start_time = time.time()  # to keep track of elapsed time

    # train
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print("-" * 10)

        # train and evaluate on validation for each epoch
        for phase in ('train', 'val'):
            if phase == 'train':
                # update the scheduler only for train, once per epoch
                scheduler.step()
                model.train()  # set model to train mode
            else:
                model.eval()  # set model to evaluation mode

            # variables for accumulating batch statistics
            running_loss = 0.0
            running_corrects = 0

            # progress bar for each epoch phase
            with tqdm(desc=phase.capitalize(), total=dataset_sizes[phase],
                      leave=False, unit="images") as pbar:
                # iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()  # reset gradients

                    # set gradient to true only for training
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backprop and update weights during train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # we multiply by input size, because loss.item() is only
                    # for a single example in our batch found link here:
                    # https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                    # update pbar with size of our batch
                    pbar.update(inputs.size(0))

            # the epoch loss is the average loss across the entire dataset
            epoch_loss = running_loss / dataset_sizes[phase]
            # accuracy is also the average of the corrects
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} ' +
                  f'Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            if phase == 'train':
                # write train loss and accuracy
                with open(results_filepath, 'a') as f:
                    f.write(f'{epoch + 1},{epoch_loss},{epoch_acc},')
            if phase == 'val':
                # write validation accuracy
                with open(results_filepath, 'a') as f:
                    f.write(f'{epoch_acc}\n')
                # deep copy the model if we perform better
                if epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training completed in {int(time_elapsed // 60)}m ' +
          f'{int(time_elapsed % 60)}s')
    print(f'Best validation accuracy: {best_accuracy:.4f}')
    # load best model weights and return the model
    model.load_state_dict(best_model_weights)
    return model


def make_results_file():
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """
    results_filepath = os.path.join(outdir, "train.csv")

    with open(results_filepath, 'w') as f:
        f.write("epoch,loss,train_acc,val_acc\n")
    return results_filepath


if __name__ == '__main__':
    main()
