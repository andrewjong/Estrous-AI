
import torch.nn as nn
import torch.optim as optim

import torchvision

from .trainable import Trainable


class Inception(Trainable):
    """Class for Inception model attributes.

    Raises:
        ValueError -- if passed attribute is not "train" or "fixed"
    """

    available_sizes = (18, 34, 50, 101, 152)

    def __init__(self, num_classes, num_layers, finetune_or_fixed):
        """ Initialize the class

        Arguments:
            finetune_or_fixed {string}
                -- whether to do transfer learning by finetuning or as a fixed
                   feature extractor. (default: "finetune")

        Raises:
            ValueError -- if finetune_or_fixed is not either "finetune" or
                "fixed"
        """
        if not any(int(num_layers) == size for size in self.available_sizes):
            raise ValueError('"num_layers" parameters must be one of the ' +
                             f'available sizes {self.available_sizes}. ' +
                             f'Received {num_layers}.')

        model = torchvision.models.inception_v3(pretrained=True)
        # get num input features of the fully connected layer
        num_features = model.fc.in_features

        if finetune_or_fixed == "finetune":
            # tack on output of 4 classes
            model.fc = nn.Linear(num_features, num_classes)
            optimizer = optim.SGD(
                model.parameters(), lr=0.001, momentum=0.9)

        elif finetune_or_fixed == "fixed":
            # stop gradient tracking in previous layers to freeze them
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(num_features, num_classes)
            optimizer = optim.SGD(
                model.fc.parameters(), lr=0.001, momentum=0.9)

        else:
            raise ValueError('"finetune_or_fixed" argument value must be ' +
                             'either "finetune" or "fixed". Received: ' +
                             f'"{finetune_or_fixed}".')

        criterion = nn.CrossEntropyLoss()

        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        # set the train hyperparameters in the super class
        super().__init__(model, criterion, optimizer, lr_scheduler)
