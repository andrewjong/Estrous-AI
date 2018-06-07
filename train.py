import argparse

# add your file here
TRAIN_MODEL_CHOICES = {
    "cnn_basic": "TODO: CNN BASIC CLASS",
    "resnet_transfer": "TODO: RESNET TRANSFER CLASS",
    "svc_transfer": "TODO: SVC CLASS",
}

# TODO: Convert these arguments into a configuration file later
parser = argparse.ArgumentParser(
    description="Train an estrous cycle cell-phase classifer model.")

parser.add_argument("model",
                    choices=list(TRAIN_MODEL_CHOICES),
                    help=f'Choose which model to use.')
parser.add_argument("-d", "--dataset",
                    default="data/lavage",
                    help='Root directory of dataset to use, with classes \
                    separated into separate subdirectoires \
                    (default: "data/lavage").')
parser.add_argument("-e", "--epochs", type=int, metavar="N",
                    default=50,
                    help='Number of epochs to train for (default: 50).')
parser.add_argument("-s", "--save",
                    default="models",
                    help='File path or directory to save model parameters \
                    (default directory: "models"). If the passed argument is \
                    a directory, the program will generate a file name. \
                    If the path does not exist, the program will create it.')
parser.add_argument("-r", "--results",
                    default="results",
                    help='Directory to store train results \
                    (default: "results"). If the directory path does not \
                    exist, the program will create it.')

args = parser.parse_args()
