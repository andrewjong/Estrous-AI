import torch.nn as nn
import torch.optim as optim

import torchvision.models


class ResNet:
    """Class for ResNet18 model attributes.

    Raises:
        ValueError -- if passed attribute is not "train" or "fixed"
    """

    available_sizes = (18, 34, 50, 101, 152)

    def __init__(self, num_layers=18, finetune_or_fixed="finetune"):
        """ Initialize the class

        Arguments:
            finetune_or_fixed {string} -- whether to do transfer learning by 
                finetuning or as a fixed feature extractor. (default: "finetune")

        Raises:
            ValueError -- if finetune_or_fixed is not either "finetune" or
                "fixed"
        """
        if not any(int(num_layers) == size for size in self.available_sizes):
            raise ValueError('"num_layers" parameters must be one of the ' +
                f'available sizes {self.available_sizes}. Received {num_layers}.')

        self.model = getattr(torchvision.models,
                             "resnet" + num_layers)(pretrained=True)
        num_features = self.model.fc.in_features  # get num input features of fc layer

        if finetune_or_fixed == "finetune":
            # tack on output of 4 classes
            self.model.fc = nn.Linear(num_features, 4)
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=0.001, momentum=0.9)

        elif finetune_or_fixed == "fixed":
            # stop gradient tracking in previous layers to freeze them
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Linear(num_features, 4)
            self.optimizer = optim.SGD(
                self.model.fc.parameters(), lr=0.001, momentum=0.9)

        else:
            raise ValueError('"finetune_or_fixed" argument value must be either ' +
                             f'"finetune" or "fixed". Received: "{finetune_or_fixed}".')

        self.criterion = nn.CrossEntropyLoss()

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1)
