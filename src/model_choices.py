""" This file  maps custom names to model classes."""
# import your model from src/
from src.cnn_resnet_transfer import ResNet

# add the custom name and imported class here
TRAIN_MODEL_CHOICES = {
    "resnet_transfer": ResNet,
}
