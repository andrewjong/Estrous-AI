import argparse
import os

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
                    "'train', 'val', or 'test' (default: 'val').")
parser.add_argument("-r", "--results_dir",
                    default="results",
                    help='Directory to store evaluation results \
                    (default: "results"). If the directory path does not \
                    exist, the program will create it.')
args = parser.parse_args()


def main():
    # get our data
    datasets, dataloaders = utils.get_datasets_and_loaders(
        args.data_dir, args.subset, include_paths=True)
    dataset = datasets[args.subset]
    dataloader = dataloaders[args.subset]

    HyperParamsClass = TRAIN_MODEL_CHOICES[args.model]
    model = HyperParamsClass(len(dataset.classes), *args.added_args).model
    model.load_state_dict(torch.load(args.model_file))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    header = ','.join(['image_name'] + dataset.classes +
                      ['predicted', 'label'])
    # print(header)
    results_file = make_results_file(header)
    print("Writing results to", results_file)

    loss = 0.0
    corrects = 0

    with tqdm(desc="Eval", total=len(dataset)) as pbar:
        # for all the inputs, make a prediction
        for (inputs, labels), paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                # make prediction by passing forward through the model
                raw_outputs = model(inputs)
                outputs = torch.nn.Softmax(dim=1)(raw_outputs)
                _, predictions = torch.max(outputs, 1)
                for i, row in enumerate(outputs):
                    # keep only the first 4 decimals
                    values_as_strings = [f'{value:.4f}'
                                         for value in row.numpy()]
                    image_name = os.path.basename(paths[i])
                    # get the actual class names instead of just indices
                    predicted_class_name = dataset.classes[predictions[i]]
                    label_class_name = dataset.classes[labels[i]]
                    # add the class names to the array
                    values_as_strings.insert(0, image_name)
                    values_as_strings.extend(
                        [predicted_class_name, label_class_name])
                    # make the array a single csv string
                    csv_line = ','.join(values_as_strings)
                    # write the file
                    with open(results_file, 'a') as f:
                        f.write(csv_line + '\n')

            pbar.update(inputs.size(0))


def make_results_file(header):
    """Creates a file (overwrites if existing) for recording train results
    using the results path specified in parse_args.
    Adds in a header for the file as "epoch,loss,train_acc,val_acc"

    Returns:
        string -- path of the created file
    """
    os.makedirs(args.results_dir, exist_ok=True)
    # get just the basename with no extension
    model_name, _ = os.path.splitext(os.path.basename(args.model_file))
    results_filepath = os.path.join(args.results_dir,
                                    "_".join([model_name, args.subset,
                                              "eval.csv"]))
    # write the csv header
    with open(results_filepath, 'w') as f:
        # TODO add image file name as first column
        f.write(header + "\n")
    return results_filepath


if __name__ == '__main__':
    main()
