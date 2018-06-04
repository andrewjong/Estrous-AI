import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

# plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ])
}

data_dir = os.path.join("..", "data", "lavage")
image_datasets = {subset: datasets.ImageFolder(os.path.join(data_dir, subset),
                                               data_transforms[subset])
                  for subset in ('train', 'val')}

dataloaders = {subset: torch.utils.data.DataLoader(
    image_datasets[subset], batch_size=4, shuffle=True, num_workers=4)
    for subset in ('train', 'val')}

dataset_sizes = {subset: len(image_datasets[subset]) for subset in ('train', 'val')}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    """Train the given model.
    
    Arguments:
        model {[type]} -- the model to train
        criterion {[type]} -- a criterion (loss function) for the optimizer
        optimizer {[type]} -- the optimizer to use such as SGD, Adam, RMSProp, etc
        scheduler {[type]} -- update the learning rate based on epochs completed
    
    Keyword Arguments:
        num_epochs {int} -- number of epochs to train for (default: {50})
    """
    since = time.time() # to keep track of elapsed time

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-" * 10)

        for phase in ('train', 'val'):
            if phase == 'train':
                # update the scheduler only for train, once per epoch
                scheduler.step()
                model.train() # set model to train mode
            else:
                model.eval() # set model to evaluation mode
            
            # variables for evaluating model accuracy
            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # reset gradients

                # forward
                # set gradient to true only for training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backprop and update weights during train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0) # what does size(0) do, and why do we multiply by it?
                    # oh, perhaps  loss.item() is a vector? so we're doing dot product perhaps?
                    # found link here: https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
                running_corrects += torch.sum(predictions == labels.data)