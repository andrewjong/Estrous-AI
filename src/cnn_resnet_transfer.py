import torch.nn as nn
import torch.optim as optim

import torchvision


class ResNet18:
    """Class for ResNet18 model attributes.

    Raises:
        ValueError -- if passed attribute is not "train" or "fixed"
    """

    def __init__(self, finetune_or_fixed):
        """ Initialize the class

        Arguments:
            finetune_or_fixed {string} -- whether to do transfer learning by 
                finetuning or as a fixed feature extractor

        Raises:
            ValueError -- if finetune_or_fixed is not either "finetune" or
                "fixed"
        """

        self.model = torchvision.models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features  # get num input features of fc layer

        if finetune_or_fixed == "finetune":
            self.model.fc = nn.Linear(num_features, 4) # tack on output of 4 classes
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
            raise ValueError('finetune_or_fixed argument must be either \
            "finetune" or "fixed". ' + f'Received "{finetune_or_fixed}".')

        self.criterion = nn.CrossEntropyLoss()

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1)
