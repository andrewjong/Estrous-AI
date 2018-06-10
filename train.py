import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

import utils
from src.cnn_resnet_transfer import ResNet
from tqdm import tqdm

# add your class here
TRAIN_MODEL_CHOICES = {
    "resnet18_transfer": ResNet,
    # "cnn_basic": "TODO CNN BASIC CLASS",
    # "svc_transfer": "TODO SVC CLASS",
}

# TODO: Convert these arguments into a configuration file later
parser = argparse.ArgumentParser(
    description="Train an estrous cycle cell-phase classifer model.")

parser.add_argument("model",
                    choices=list(TRAIN_MODEL_CHOICES),
                    help=f'Choose which model to use.')
parser.add_argument("-d", "--data_dir",
                    default="data/lavage",
                    help='Root directory of dataset to use, with classes \
                    separated into separate subdirectoires \
                    (default: "data/lavage").')
parser.add_argument("-e", "--epochs", type=int, metavar="N",
                    default=50,
                    help='Number of epochs to train for (default: 50).')
parser.add_argument("-s", "--save_dir",
                    default="models",
                    help='File path or directory to save model parameters \
                    (default directory: "models"). If the passed argument is \
                    a directory, the program will generate a file name. \
                    If the path does not exist, the program will create it.')
parser.add_argument("-r", "--results_dir",
                    default="results",
                    help='Directory to store train results \
                    (default: "results"). If the directory path does not \
                    exist, the program will create it.')
parser.add_argument("-a", "--added_args", nargs="+",
                    default=[],
                    help="Pass addiitional arguments for instantiating the \
                    chosen model.")

args = parser.parse_args()


def main():
    # instantiate the model object
    chosen_class = TRAIN_MODEL_CHOICES[args.model]
    chosen = chosen_class(*args.added_args)

    # obtain train and validation datasets and dataloaders
    datasets, dataloaders = utils.get_datasets_and_loaders(
        args.data_dir, "train", "val")
    dataset_sizes = {subset: len(datasets[subset])
                     for subset in ('train', 'val')}

    # make results dir path
    os.makedirs(args.results_dir, exist_ok=True)
    # train
    trained_model = train_model(
        chosen.model, chosen.criterion, chosen.optimizer,
        chosen.lr_scheduler, args.epochs,
        dataloaders, dataset_sizes)

    # Save the model
    os.makedirs(args.save_dir, exist_ok=True)
    save_fname = make_model_fname() + ".pth"
    save_path = os.path.join(args.save_dir, save_fname)
    torch.save(trained_model.state_dict(), save_path)


def train_model(model, criterion, optimizer, scheduler, num_epochs,
                dataloaders, dataset_sizes):
    """Train the given model.

    Arguments:
        model {[type]} -- the model to train
        criterion {[type]} -- a criterion (loss function) for the optimizer
        optimizer {[type]} -- the optimizer to use such as SGD, Adam, RMSProp, etc
        scheduler {[type]} -- update the learning rate based on epochs completed

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
                scheduler.step()  # update the scheduler only for train, once per epoch
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

                    # we multiply by input size, because loss.item() is only for a single example in our batch
                    # found link here: https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                    # update pbar with size of our batch
                    pbar.update(inputs.size(0))

            # the epoch loss is the average loss across the entire dataset
            epoch_loss = running_loss / dataset_sizes[phase]
            # accuracy is also the average of the corrects
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(
                f'{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

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
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training completed in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    # load best model weights and return the model
    model.load_state_dict(best_model_weights)
    return model

def make_model_fname():
    """Generate a file name that includes the model name, plus any additional
    args passed to instantiate the model object, separated by underscores.
    
    Returns:
        string -- formatted as "[model]_[additional_args]"
    """

    return "_".join([args.model] + args.added_args) 


def make_results_file():
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """

    results_filepath = os.path.join(
        args.results_dir, make_model_fname() + "_train.csv")

    with open(results_filepath, 'w') as f:
        f.write("epoch,loss,train_acc,val_acc\n")
    return results_filepath


if __name__ == '__main__':
    main()
