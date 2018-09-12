import copy
import gc
import json
import os
import signal
import sys
import time

import torch
import src.utils as utils
from tqdm import tqdm

from common_constants import META_FNAME, MODEL_PARAMS_FNAME, TRAIN_RESULTS_FNAME


class Trainable:
    """Base class for a trainable neural network architecture.
    Contains the model, criterion, optimizer, and scheduler as attributes.
    Also includes a train function.

    All neural network model choices should extend this class.
    """

    def __init__(self, datadir, model, criterion, optimizer, lr_scheduler=None,
                 batch_size=4, outdir=None):
        """Initialize common train hyperparameters. These hyperparameters must
        be set

        Arguments:
            model {torch.nn.Module} -- model structure
            criterion {torch.nn._Loss} -- loss function
            optimizer {torch.optim.Optimizer} -- optimizer, e.g. SGD or Adam
            lr_scheduler {torch.optim._LRScheduler} -- learning rate scheduler
        """

        self.datadir = datadir
        self.dataloaders = self.make_dataloaders(datadir, model, batch_size)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def make_dataloaders(datadir, model, batch_size):
        model_name = str(model)[1]
        image_size = utils.determine_image_size(model_name)
        return utils.get_dataloaders(datadir, ('train', 'val'), image_size=image_size,
                                     batch_size=batch_size)

    def handle_interrupt(self, signal, frame):
        if not self.interrupted:
            self.interrupted = True
            print(
                "Sigint caught!\nTraining will stop after this epoch and the "
                + "best model so far will be saved.\nOR press Ctrl-C again to quit "
                + "immediately without saving."
            )
        else:
            print("Stopping...")
            sys.exit(1)

    def save(self, outdir=None, extra_meta=None):
        """Saves the model weights (via state dict) and meta info about how the
         model was trained to the specified output directory.

        Arguments:
            outdir {string} -- directory to output files

        Keyword Arguments:
            extra_meta {dict} -- extra information to put in the meta file
            (default: {None})
        """
        model_file = os.path.join(outdir, MODEL_PARAMS_FNAME)
        torch.save(self.model.state_dict(), model_file)
        meta_dict = {
            "best_val_accuracy": self.best_val_accuracy,
            "train_accuracy": self.associated_train_accuracy,
            "train_loss": self.associated_train_loss,
            "finished_epochs": self.finished_epochs,
        }
        if extra_meta:
            meta_dict = extra_meta.update(meta_dict)

        meta_out = os.path.join(outdir, META_FNAME)
        with open(meta_out, "w") as out:
            json.dump(meta_dict, out, indent=4)

    def _prepare_results_file(self):
        """Creates a file (overwrites if existing) for recording train results
        using the results path specified in parse_args.
        Adds in a header for the file as "epoch,loss,train_acc,val_acc"

        Returns:
            string -- path of the created file
        """
        results_filepath = os.path.join(self.outdir, TRAIN_RESULTS_FNAME)
        header = "steps,train_loss,train_acc,val_acc"
        results_filepath = utils.make_csv_with_header(results_filepath, header)
        self.results_filepath = results_filepath
        return results_filepath

    def _run_through_images():

    def train(self, num_epochs, early_stop_limit=None, verbose=True):
        """Train the model based on the instance's attributes, as well as the
        passed in arguments. Automatically moves to GPU if available.

        Arguments:
            dataloaders {(DataLoader, DataLoader)}
                -- train set dataloader, validation set dataloader
            num_epochs {int} -- number of epochs to trian for
            (default: None)
            early_stop {int} -- stop training early if validation accuracy does
            not improve for {early_stop} epochs. potentially saves time
            (default: 6)

        Returns:
            float -- best validation accuracy of the model
        """
        # ================== SETUP ==================
        model, criterion, optimizer, scheduler, dataloaders = (  # naming convenience
            self.model, self.criterion, self.optimizer,
            self.lr_scheduler, self.dataloaders
        )
        dataset_sizes = {  # sizes for our progress bar
            phase: len(loader.dataset)
            for phase, loader in dataloaders.items()
        }

        # choose the best compute device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # setup our best results to return later
        best_model_weights = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0

        if early_stop_limit:
            early_stop_counter = 0  # for stopping training early
        results_filepath = self._prepare_results_file()  # for writing epoch results
        start_time = time.time()  # to keep track of elapsed time

        # ================== TRAIN LOOP ==================
        # for each epoch, train on the train set then evalute on the val set
        for epoch in range(num_epochs):
            # print overall progress
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print("-" * 10)

            # during the epoch, run both train and evaluation
            for phase in ("train", "val"):
                if phase == "train":
                    # update the learning rate scheduler only for train, once per epoch
                    scheduler.step() if scheduler else None
                    model.train()  # set model to train mode
                else:
                    model.eval()  # set model to evaluation mode

                # variables for accumulating batch statistics
                running_loss = 0.0
                running_corrects = 0

                # progress bar for each epoch phase
                description = f"Epoch {epoch + 1}, {phase.capitalize()}"
                # add best val to progress bar if we're in non-verbose mode
                if not verbose:
                    description += f", best val={best_val_acc:4f}"
                with tqdm(desc=description, total=dataset_sizes[phase], leave=False,
                          unit="images",) as pbar:

                    # iterate over data
                    for inputs, labels in dataloaders[phase]:
                        # place on device for computation
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()  # reset gradients

                        # set gradient to true only for training
                        with torch.set_grad_enabled(phase == "train"):
                            # forward
                            outputs = model(inputs)
                            _, predictions = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            # backprop and update weights during train
                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                        # we multiply by input size, because loss.item() is
                        # only for a single example in our batch.
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(predictions ==
                                                      labels.data)

                        # update pbar with size of our batch
                        pbar.update(inputs.size(0))

                # the epoch loss is the average loss across the entire dataset
                phase_loss = running_loss / dataset_sizes[phase]
                # accuracy is also the average of the corrects
                phase_acc = running_corrects.double() / dataset_sizes[phase]
                # print results for this epoch phase
                if verbose:
                    print(f"{phase.capitalize()} "
                          + f"Loss: {phase_loss:.4f}, Acc: {phase_acc:.4f}")

                # WRITE WHAT HAPPENED
                if phase == "train":
                    batch_size = dataloaders['train'].batch_size
                    step_num = int(
                        (epoch + 1) * dataset_sizes["train"] / batch_size)
                    # store these for when we do validation
                    train_acc = phase_acc
                    train_loss = phase_loss
                    # write train loss and accuracy
                    if results_filepath:
                        with open(results_filepath, "a") as f:
                            f.write(f"{step_num},{phase_loss},{phase_acc},")
                elif phase == "val":
                    # write validation accuracy
                    if results_filepath:
                        with open(results_filepath, "a") as f:
                            f.write(f"{phase_acc}\n")

                    # IF we did better, update our saved best
                    if phase_acc > best_val_acc:
                        best_val_acc = phase_acc
                        associated_train_acc = train_acc
                        associated_train_loss = train_loss
                        best_model_weights = copy.deepcopy(model.state_dict())
                        if early_stop_limit:  # reset early stop counter
                            early_stop_counter = 0
                    # ELSE update our early stop counter
                    elif early_stop_limit:
                        if verbose:
                            remaining = early_stop_limit - early_stop_counter
                            print('Model did not perform better.' +
                                  f'Remaining tries: {remaining}/{early_stop_limit}')
                        if early_stop_counter >= early_stop_limit:
                            self.interrupted = True

            print() if verbose else None  # spacer between epochs
            self.finished_epochs = epoch + 1
            if self.interrupted:
                print("Training stopped early at",
                      self.finished_epochs, "epochs.")
                break
        gc.collect()  # trigger the gargabe collector to free up memory

        # Print summary results
        time_elapsed = time.time() - start_time
        print(f"Training completed in {int(time_elapsed // 60)}m "
              + f"{int(time_elapsed % 60)}s")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Associated train accuracy: {associated_train_acc:.4f}")
        print(f"Associated train loss: {associated_train_loss:.4f}")

        # load best model weights
        model.load_state_dict(best_model_weights)
        # Store best scores, convert torch tensor to float
        self.best_val_accuracy = float(best_val_acc)
        self.associated_train_accuracy = float(associated_train_acc)
        self.associated_train_loss = float(associated_train_loss)

        return best_val_acc
