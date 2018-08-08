import argparse

from predict import (load_model, get_meta_dict_from_load_dir,
                     get_subset_dataset_and_loader, get_device)


GROUP1 = ("proestrus", "estrus", "metestrus")
GROUP2 = ("diestrus")

def run_multistage(data_dir, subset, model_paths):
    dataset_4 = get_subset_dataset_and_loader(data_dir, subset)
    dataset_binarized = regroup_dataset_classes(dataset_4, (GROUP1, GROUP2))

    binary_model_path = model_paths[0]
    model_dir = os.path.dirname(binary_model_path)
    meta_dict = get_meta_dict_from_load_dir(model_dir)

    num_classes = len(dataset_binarized)
    device = get_device()



    model = load_model(model_path, meta_dict, num_classes, device)
    pass

def regroup_dataset_classes(dataset, groups):
    """[summary]
    
    Arguments:
        dataset {[type]} -- [description]
        groups {tuple} -- Class groups, according to how they're named in 
        dataset.classes
    """
    # We want to change all 4 of the below variables
    original_classes = dataset.classes # the class names
    original_class_to_idx = dataset.class_to_idx # the class names to number values
    original_samples = dataset.samples # a list of tuple(image paths, targets)
    original_targets = dataset.targets # just a list of targets

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
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

    run_multistage(args.data_dir, args.subset, args.models)
