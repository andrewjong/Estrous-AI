
import argparse

parser = argparse.ArgumentParser(
    description="Evaluate a model on the test set. Only do this once you're \
    sure!")

parser.add_argument("model_file",
                    help='File from which to load model parameters, typically \
                    ends with a ".pth" extension.')
parser.add_argument("-d", "--dataset",
                    default="data/lavage",
                    help='Root directory of dataset to use, with classes \
                    separated into separate subdirectoires \
                    (default: "data/lavage").')
parser.add_argument("-r", "--results",
                    default="results",
                    help='Directory to store evaluation results \
                    (default: "results"). If the directory path does not \
                    exist, the program will create it.')

args = parser.parse_args()