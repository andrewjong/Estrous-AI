import argparse

import torch

import torchvision
from tqdm import tqdm
import utils
from src.cnn_resnet_transfer import ResNet

parser = argparse.ArgumentParser(
    description="Evaluate a model on the test set. Only do this once you're \
    sure!")

# parser.add_argument("model_class")
parser.add_argument("model_file",
                    help='File from which to load model parameters, typically \
                    ends with a ".pth" extension.')
parser.add_argument("-d", "--data_dir",
                    default="data/lavage",
                    help='Root directory of dataset to use, with classes \
                    separated into separate subdirectories \
                    (default: "data/lavage").')
parser.add_argument("-s", "--subset", default="val",
                    help="Which subset of the dataset to evaluate on. E.g. " +
                    "'train', 'val', or 'test'.")
parser.add_argument("-r", "--results",
                    default="results",
                    help='Directory to store evaluation results \
                    (default: "results"). If the directory path does not \
                    exist, the program will create it.')

args = parser.parse_args()


def main():
    datasets, dataloaders = utils.get_datasets_and_loaders(
        args.data_dir, args.subset)
    dataset = datasets[args.subset]
    dataloader = dataloaders[args.subset]

    model = ResNet(num_classes=4).model
    model.load_state_dict(torch.load(args.model_file))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loss = 0.0
    corrects = 0

    with tqdm(desc="Eval", total=len(dataset)):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                print('=' * 20)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                # Note: we're forced to use .view() instead of .t() to transpose because of some weird underlying C error in pytorch:
                # RuntimeError: invalid argument 2: out of range at /opt/conda/conda-bld/pytorch-cpu_1524582300956/work/aten/src/TH/generic/THTensor.cpp:455
                # Turn row into a vector of type float
                predictions = predictions.float().view(4, 1)
                labels = labels.float().view(4, 1)
                result_matrix = torch.cat((outputs, predictions, labels), 1)
                print('result:', result_matrix)
                print('=' * 20)


if __name__ == '__main__':
    main()
