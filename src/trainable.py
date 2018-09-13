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

from common_constants import (META_FNAME, MODEL_PARAMS_FNAME, TRAIN_RESULTS_FNAME,
                              PREDICT_RESULTS_FNAME)


class Trainable:
    """Base class for a trainable neural network architecture.
    Contains the model, criterion, optimizer, and lr_scheduler as attributes.
    Also includes a train function.

    All neural network model choices should extend this class.
    """

    def __init__(self, dataloaders, model, criterion, optimizer, lr_scheduler=None,
                 outdir=None):
        """Initialize common train hyperparameters. These hyperparameters must
        be set

        Arguments:
            model {torch.nn.Module} -- model structure
            criterion {torch.nn._Loss} -- loss function
            optimizer {torch.optim.Optimizer} -- optimizer, e.g. SGD or Adam
            lr_scheduler {torch.optim._LRScheduler} -- learning rate scheduler
        """

        self.dataloaders = dataloaders
        self.dataset_sizes = utils.get_dataset_sizes(self.dataloaders)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.outdir = outdir
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def load_model_weights(self, weights_file):
        """Load in weights to the model from a file

        Arguments:
            weights_file {String} -- path to model weights (*.pth )
        """

        pretrained_weights_dict = torch.load(
            weights_file, map_location=lambda storage, loc: storage)

        model_dict = self.model.state_dict()
        excluded = ['fc.weight', 'fc.bias', 'last_linear.weight']
        pretrained_weights_dict = {
            k: v
            for k, v in pretrained_weights_dict.items() if k not in excluded
        }
        model_dict.update(pretrained_weights_dict)
        self.model.load_state_dict(model_dict)

    def save(self, extra_meta=None):
        """Saves the model weights (via state dict) and meta info about how the
         model was trained to the specified output directory.

        Arguments:
            outdir {string} -- directory to output files

        Keyword Arguments:
            extra_meta {dict} -- extra information to put in the meta file
            (default: {None})
        """
        model_file = os.path.join(self.outdir, MODEL_PARAMS_FNAME)
        torch.save(self.model.state_dict(), model_file)
        meta_dict = {
            "best_val_accuracy": self.best_val_accuracy,
            "train_accuracy": self.associated_train_accuracy,
            "train_loss": self.associated_train_loss,
            "finished_epochs": self.finished_epochs,
        }
        if extra_meta:
            meta_dict = extra_meta.update(meta_dict)

        meta_out = os.path.join(self.outdir, META_FNAME)
        with open(meta_out, "w") as out:
            json.dump(meta_dict, out, indent=4)

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
        self._train_setup(early_stop_limit, verbose)

        # setup our best results to return later
        best_model_weights = copy.deepcopy(self.model.state_dict())

        # ================== TRAIN LOOP ==================
        # for each epoch, train on the train set then evalute on the val set
        for epoch in range(num_epochs):
            self._print_epoch_header(epoch, num_epochs)

            # during the epoch, run both train and evaluation
            for subset in ("train", "val"):
                # perform subset action on images
                subset_loss, subset_acc = self._run_through_images(epoch, subset)

                # WRITE WHAT HAPPENED
                if subset == "train":
                    step_num = int((epoch + 1) *
                                   self.dataset_sizes["train"] /
                                   self.dataloaders['train'].batch_size)
                    # write train loss and accuracy
                    if self.results_filepath:
                        with open(self.results_filepath, "a") as f:
                            f.write(f"{step_num},{subset_loss},{subset_acc},")
                    # store these for when we do validation
                    train_acc, train_loss = subset_acc, subset_loss

                elif subset == "val":
                    # write validation accuracy
                    if self.results_filepath:
                        with open(self.results_filepath, "a") as f:
                            f.write(f"{subset_acc}\n")

                    # IF we did better, update our saved best
                    if subset_acc > self.best_val_accuracy:
                        self._store_best(subset_acc, train_acc, train_loss)
                        best_model_weights = copy.deepcopy(
                            self.model.state_dict())
                        if self.early_stop_limit:  # reset early stop counter
                            self.early_stop_counter = 0
                    # ELSE update our early stop counter
                    elif self.early_stop_limit:
                        self._increment_stop_limit()

            print() if self._verbose else None  # spacer between epochs
            self.finished_epochs = epoch + 1
            if self.interrupted:
                print("Training stopped early at",
                      self.finished_epochs, "epochs.")
                break

        # load best model weights
        self.model.load_state_dict(best_model_weights)

        # Print summary results
        self._print_train_summary()

        return self.best_val_accuracy

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

    def _prepare_train_results(self):
        """Creates a file (overwrites if existing) for recording train results
        using the results path specified in parse_args.
        Adds in a header for the file as "epoch,loss,train_acc,val_acc"

        Returns:
            string -- path of the created file
        """
        if self.outdir is None:
            return None
        results_filepath = os.path.join(self.outdir, TRAIN_RESULTS_FNAME)
        header = "steps,train_loss,train_acc,val_acc"
        results_filepath = utils.make_csv_with_header(results_filepath, header)
        return results_filepath

    def _run_through_images(self, epoch, subset):
        if subset == "train":
            # update the learning rate scheduler only for train, once per epoch
            self.lr_scheduler.step() if self.lr_scheduler else None
            self.model.train()  # set model to train mode
        else:
            self.model.eval()  # set model to evaluation mode

        # variables for accumulating batch statistics
        running_loss = 0.0
        running_corrects = 0

        description = self._make_pbar_description(epoch, subset)
        # progress bar for each epoch subset
        with tqdm(desc=description, total=self.dataset_sizes[subset], leave=False,
                  unit="images",) as pbar:

            # iterate over data
            for inputs, labels in self.dataloaders[subset]:
                # place on device for computation
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # reset gradients

                # set gradient to true only for training
                with torch.set_grad_enabled(subset == "train"):
                    # forward
                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    # backprop and update weights during train
                    if subset == "train":
                        loss.backward()
                        self.optimizer.step()

                # we multiply by input size, because loss.item() is
                # only for a single example in our batch.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions ==
                                              labels.data)

                # update pbar with size of our batch
                pbar.update(inputs.size(0))

        # the epoch loss is the average loss across the entire dataset
        subset_loss = running_loss / self.dataset_sizes[subset]
        # accuracy is also the average of the corrects
        subset_acc = running_corrects.double() / self.dataset_sizes[subset]
        # print results for this epoch subset
        if self._verbose:
            print(f"{subset.capitalize()} "
                  + f"Loss: {subset_loss:.4f}, Acc: {subset_acc:.4f}")

        return subset_loss, subset_acc

    def _make_pbar_description(self, epoch, subset):
        description = f"Epoch {epoch + 1}, {subset.capitalize()}"
        # add best val to progress bar if we're in non-verbose mode
        if self._verbose:
            description += f', best val={self.best_val_accuracy:4f}'
        return description

    def _store_best(self, val_acc, train_acc, train_loss):
        self.best_val_accuracy = val_acc
        self.associated_train_accuracy = train_acc
        self.associated_train_loss = train_loss

    def _increment_stop_limit(self):
        if self._verbose:
            remaining = self.early_stop_limit - self.early_stop_counter
            print('Model did not perform better.' +
                  f'Remaining tries: {remaining}/{self.early_stop_limit}')
        if self.early_stop_counter >= self.early_stop_limit:
            self.interrupted = True

    def _print_epoch_header(self, epoch, num_epochs):
        # print overall progress
        if self._verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

    def _train_setup(self, early_stop_limit, verbose):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # choose the best compute device
        self.model = self.model.to(self.device)
        self.best_val_accuracy = 0.0

        # for stopping training early
        self.early_stop_limit = early_stop_limit if early_stop_limit else None
        self.early_stop_counter = 0 if early_stop_limit else None

        self._verbose = verbose

        self.results_filepath = self._prepare_train_results()  # for writing epoch results
        self.start_time = time.time()  # to keep track of elapsed time
        pass

    def _print_train_summary(self):
        self.end_time = time.time()
        self.train_time = self.end_time - self.start_time
        print(f"Training completed in {int(self.train_time // 60)}m "
              + f"{int(self.train_time % 60)}s")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Associated train accuracy: {self.associated_train_acc:.4f}")
        print(f"Associated train loss: {self.associated_train_loss:.4f}")
