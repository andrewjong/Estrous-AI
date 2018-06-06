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

# Normalization parameters
MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        # data augmentation, randomly flip and vertically flip across epochs
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # convert to PyTorch tensor
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(MEANS, STDS)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])
}

data_dir = os.path.join("..", "data", "lavage")
image_datasets = {subset: datasets.ImageFolder(os.path.join(data_dir, subset),
                                               data_transforms[subset])
                  for subset in ('train', 'val')}

dataloaders = {subset: torch.utils.data.DataLoader(
    image_datasets[subset], batch_size=4, shuffle=True, num_workers=4)
    for subset in ('train', 'val')}

dataset_sizes = {subset: len(image_datasets[subset])
                 for subset in ('train', 'val')}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inputs, title=None):
    """ Show a PyTorch Tensor as an image, after unnormalizing """
    inputs = inputs.numpy().transpose((1, 2, 0))
    mean = np.array(MEANS)
    std = np.array(STDS)
    # unnormalize
    inputs = std * inputs + mean
    # constrain values to be between 0 and 1, in case any went over
    inputs = np.clip(inputs, 0, 1)
    plt.imshow(inputs)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # let plots update


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
    since = time.time()  # to keep track of elapsed time

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-" * 10)

        for phase in ('train', 'val'):
            if phase == 'train':
                # update the scheduler only for train, once per epoch
                scheduler.step()
                model.train()  # set model to train mode
            else:
                model.eval()  # set model to evaluation mode

            # variables for evaluating model accuracy
            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # reset gradients

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

                # found link here: https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
                # we multiply by input size, because loss.item() is only for a single example in our batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            # the epoch loss is the average loss across the entire dataset
            epoch_loss = running_loss / dataset_sizes[phase]
            # accuracy is also the average of the corrects
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # deep copy the model if we perform better
            if phase == 'val' and epoch_acc > best_accuracy:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    # load best model weights and return the model
    model.load_state_dict(best_model_weights)
    return model


def visualize_model(model, num_images=6):
    """Visualize the model's current predictions on some images.

    Arguments:
        model {[type]} -- [description]

    Keyword Arguments:
        num_images {int} -- [description] (default: {6})
    """

    was_training = model.training
    # why do we do this?
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[predictions[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

            model.train(mode=was_training)


def train_by_finetune():
    """Train by initializing parameters to ResNet18, then by tuning the parameters
    for our specific problem.
    """

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features  # get num input features of fc layer
    model.fc = nn.Linear(num_features, 4) # set the output to 4 for 4 classes


    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler)

def train_by_fixed_feature_extractor():
    """Train by freezing all but the last layer.
    """

    model = models.resnet18(pretrained=True)

    # stop gradient tracking in previous layers to freeze them
    for param in model.parameters():
        param.requires_grad = False

    
    num_features = model.fc.in_features  # get num input features of fc layer
    model.fc = nn.Linear(num_features, 4) # set the output to 4 for 4 classes


    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler)


    