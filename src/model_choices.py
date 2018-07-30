""" This file  maps custom names to model classes."""
# place your model file in src/
from .model_resnet_transfer import ResNet

# add the custom name and imported class here
TRAIN_MODEL_CHOICES = {
    "resnet_transfer": ResNet,
}
