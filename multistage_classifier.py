import argparse
import copy
import os

import torch
from tqdm import tqdm

from predict import (get_device, get_meta_dict_from_load_dir,
                     get_subset_dataset_and_loader, load_model,
                     prepare_results_csv, write_row_prediction)
from src.utils import SORT_BY_PHASE_FN

device = get_device()
num_gpus = torch.cuda.device_count()


def main(data_dir, subset, model_paths, out_dir):
    print("Preparing setup...")
    (dataset, dataloader, binary_model, trinary_model) = setup_multistage(
        data_dir, subset, model_paths, out_dir)

    # print("Classes:", dataset.classes)
    # quit()
    print("Running predictions...")
    results_file = run_twostep_classifier(dataset, subset, dataloader,
                                          binary_model, trinary_model)


def setup_multistage(data_dir, subset, model_paths, out_dir):
    print("Loading dataset")
    quartary_dataset, quartary_dataloader = get_subset_dataset_and_loader(
        "data/4_class_11", "val", batch_size=1)

    print("Loading binary model")
    binary_model_path = model_paths[0]
    binary_meta_dict = get_meta_dict_from_model_path(binary_model_path)
    binary_model = load_model(binary_model_path, binary_meta_dict, 2, device)

    print("Loading trinary model")
    trinary_model_path = model_paths[1]
    trinary_meta_dict = get_meta_dict_from_model_path(trinary_model_path)
    trinary_model = load_model(trinary_model_path, trinary_meta_dict, 3,
                               device)

    return quartary_dataset, quartary_dataloader, binary_model, trinary_model


# TODO: THIS CODE IS VERY BRITTLE. WE MUST FIX IT. Brittle because rushing the
# final-week deadline
# too many hardcoded numbers
# for example, prediction of diestrus == 0
# also the +1 offset for trinary output
def run_twostep_classifier(dataset, subset, dataloader, binary_model,
                           trinary_model):
    # get where we will write results to
    class_names = [
        "diestrus", "not_diestrus", "proestrus", "estrus", "metestrus"
    ]
    results_file = prepare_results_csv(out_dir, subset, class_names)

    print("Writing results to", results_file)
    with tqdm(desc="Twostep", total=len(dataset)) as pbar:
        for input_tensor, label, (path, ) in dataloader:
            input_tensor = input_tensor.to(device)
            label = label.to(device)

            # step 1
            with torch.set_grad_enabled(False):
                binary_output = torch.nn.Softmax(dim=1)(
                    binary_model(input_tensor))

            _, prediction_1 = torch.max(binary_output, 1)

            # prediction 1 means diestrus
            if prediction_1 == 0:
                predicted_class = "diestrus"
                trinary_output = torch.zeros(
                    1, 3, device=device)  # filler output

            # step 2, only if needed
            elif prediction_1 == 1:
                with torch.set_grad_enabled(False):
                    trinary_output = torch.nn.Softmax(dim=1)(
                        trinary_model(input_tensor))
                _, prediction_2 = torch.max(trinary_output, 1)

                # offset by 1 because trinary output lacks diestrus
                predicted_class = dataset.classes[prediction_2 + 1]
            else:
                raise ValueError(
                    "Some shit happened, first prediction was not binary.")

            image_name = os.path.basename(path)
            label_class = dataset.classes[label]

            # print("binary", torch.t(binary_output))
            # print("trinary", torch.t(trinary_output))
            row = torch.cat((binary_output, trinary_output), dim=1).view(5)
            # print("row", row.cpu().numpy())

            write_row_prediction(image_name, row, label_class, predicted_class,
                                 results_file)

            pbar.update(1)


def get_meta_dict_from_model_path(model_path):
    parent_dir = os.path.dirname(model_path)
    meta_dict = get_meta_dict_from_load_dir(parent_dir)
    return meta_dict


# TURNS OUT WE DON'T NEED THIS. USEFUL CODE THOUGH, SAVE IT AS A GITHUB GIST
def regroup_dataset_classes(dataset, groups):
    """Regroup a pytorch dataset 

    New class indices will follow the order the groups are passed in.

    Arguments:
        dataset {torchvision.datasets} -- [description]
        groups {tuple} -- Class groups, according to how they're named in
        dataset.classes
    """
    dataset = copy.deepcopy(dataset)  # do not mutate the original
    # Argument checking
    check_valid_classes_in_groups(dataset.classes, groups)

    # We want to change all 4 of the below variables
    # the class names to numerical target values
    # a list of tuple(image paths, targets)
    # just a list of targets
    old_class_to_idx, old_samples = (dataset.class_to_idx, dataset.samples)

    # new class names are originals joined by underscore
    dataset.classes = [
        "_".join(sorted(group, key=SORT_BY_PHASE_FN)) for group in groups
    ]

    dataset.class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(dataset.classes)
    }

    # mapping for making new indices
    old_idx_to_new_idx = {
        old_class_to_idx[name]: new_idx
        for new_idx, group in enumerate(groups) for name in group
    }
    # unzip
    image_paths, old_targets = tuple(zip(*old_samples))

    dataset.targets = [old_idx_to_new_idx[i] for i in old_targets]
    dataset.samples = list(zip(image_paths, dataset.targets))

    return dataset


def check_valid_classes_in_groups(original_classes, groups):
    """Check that all the classes are accounted for.

    Arguments:
        original_classes {[type]} -- [description]
        groups {[type]} -- [description]

    Raises:
        ValueError -- [description]

    Returns:
        [type] -- [description]
    """

    flattened = [item for group in groups for item in group]
    valid = sorted(original_classes) == sorted(flattened)
    if not valid:
        raise ValueError(f'Classes passed do not match')
    return valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-e",
        "--experiment_name",
        default='multistage_experiment',
        help='Name of the experiment. This becomes the \
                        subdirectory that the experiment output is stored in, \
                        i.e. "experiments/my_experiment/" (default: \
                        "multistage_experiment").')
    parser.add_argument(
        "-d",
        "--data_dir",
        default=False,
        help="Directory of the 4-CLASS dataset to classify. This script only \
        works with 4-class dataset input.")
    parser.add_argument(
        "models",
        nargs="+",
        help="The models, in sequential \
        order, to use in the multistage classification pipeline.")
    parser.add_argument(
        "-s",
        "--subset",
        default="val",
        help="Which subset of the dataset to evaluate on. " +
        "E.g. 'train', 'val', or 'test' (default: 'val').")
    args = parser.parse_args()

    out_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    main(args.data_dir, args.subset, args.models, out_dir)
