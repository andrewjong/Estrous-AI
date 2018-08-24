import copy
import signal
import sys
import time

import torch
from tqdm import tqdm

interrupted = False


def signal_handler(signal, frame):
    global interrupted
    if not interrupted:
        interrupted = True
        print("Sigint caught!\nTraining will stop after this epoch and the " +
              "best model so far will be saved.\nOR press Ctrl-C again to quit " +
              "immediately without saving.")
    else:
        print("Stopping...")
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


class Trainable:
    """Base class for a trainable neural network architecture.
    Contains the model, criterion, optimizer, and scheduler as attributes.
    Also includes a train function.

    All neural network model choices should extend this class.
    """

    def __init__(self, model, criterion, optimizer, lr_scheduler=None):
        """Initialize common train hyperparameters. These hyperparameters must
        be set

        Arguments:
            model {torch.nn.Module} -- model structure
            criterion {torch.nn._Loss} -- loss function
            optimizer {torch.optim.Optimizer} -- optimizer, e.g. SGD or Adam
            lr_scheduler {torch.optim._LRScheduler} -- learning rate scheduler
        """

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self, dataloaders, dataset_sizes, num_epochs, results_filepath):
        """Train the model based on the instance's attributes, as well as the
        passed in arguments. Automatically moves to GPU if available.

        Arguments:
            dataloaders {(DataLoader, DataLoader)}
                -- train set dataloader, validation set dataloader
            dataset_sizes {(int, int)} -- train set size, validation set size
            num_epochs {int} -- number of epochs to trian for
            results_filepath {string} -- results filepath to output results to

        Returns:
            [type] -- [description]
        """
        model, criterion, optimizer, scheduler = (self.model, self.criterion,
                                                  self.optimizer,
                                                  self.lr_scheduler)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # setup our best results to return later
        best_model_weights = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0

        start_time = time.time()  # to keep track of elapsed time

        # Train
        # summary: for each epoch, train on the train set, then get
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print("-" * 10)

            # during the epoch, run both train and evaluation
            for phase in ('train', 'val'):
                if phase == 'train':
                    # update the scheduler only for train, once per epoch
                    scheduler.step() if scheduler else None
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

                        # we multiply by input size, because loss.item() is
                        # only for a single example in our batch.
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(predictions ==
                                                      labels.data)

                        # update pbar with size of our batch
                        pbar.update(inputs.size(0))

                # the epoch loss is the average loss across the entire dataset
                epoch_loss = running_loss / dataset_sizes[phase]
                # accuracy is also the average of the corrects
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # print results for this epoch phase
                print(f'{phase.capitalize()} ' +
                      f'Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    batch_size = inputs.size(0)
                    step_num = int((epoch + 1) *
                                   dataset_sizes['train'] / batch_size)
                    epoch_train_acc = epoch_acc
                    epoch_train_loss = epoch_loss
                    # write train loss and accuracy
                    with open(results_filepath, 'a') as f:
                        f.write(f'{step_num},{epoch_loss},{epoch_acc},')
                elif phase == 'val':
                    # write validation accuracy
                    with open(results_filepath, 'a') as f:
                        f.write(f'{epoch_acc}\n')
                    # deep copy the model if we perform better
                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                        associated_train_acc = epoch_train_acc
                        associated_train_loss = epoch_train_loss
                        best_model_weights = copy.deepcopy(model.state_dict())

            print()  # spacer between epochs
            self.finished_epochs = epoch + 1
            if interrupted:
                print("Training stopped early at",
                      self.finished_epochs, "epochs.")
                break

        # Print summary results
        time_elapsed = time.time() - start_time
        print(f'Training completed in {int(time_elapsed // 60)}m ' +
              f'{int(time_elapsed % 60)}s')
        print(f'Best validation accuracy: {best_val_acc:.4f}')
        print(f'Associated train accuracy: {associated_train_acc:.4f}')
        print(f'Associated train loss: {associated_train_loss:.4f}')

        # Store best scores, convert torch tensor to float
        self.best_val_accuracy = float(best_val_acc)
        self.associated_train_accuracy = float(associated_train_acc)
        self.associated_train_loss = float(associated_train_loss)

        # load best model weights and return the model
        model.load_state_dict(best_model_weights)
        return model
