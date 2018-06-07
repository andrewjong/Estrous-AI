import argparse

parser = argparse.ArgumentParser(
    description="Use a model to perform inference on new unlabeled images.")
parser.add_argument("model_file",
                    help='File from which to load model parameters, typically \
                    ends with a ".pth" extension.')
parser.add_argument("inference_dir",
                    help="Directory containing images to perform inference on.")
parser.add_argument("-r", "--results",
                    default="results",
                    help='Directory to store inference results \
                    (default: "results"). If the directory path does not \
                    exist, the program will create it.')


args = parser.parse_args()
