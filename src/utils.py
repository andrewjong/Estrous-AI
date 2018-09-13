import os

import torch

import pretrainedmodels
import src.custom_models
import torchvision
from common_constants import PHASE_ORDER
from common_constants import MODEL_TO_IMAGE_SIZE, DEFAULT_IMAGE_SIZE
from strconv import convert
from torchvision import datasets, transforms


def SORT_BY_PHASE_FN(item):
    try:
        return PHASE_ORDER[item[0].lower()]
    except (KeyError):
        return ord(item[0])  # THIS IS BAD CODE


# class_names = image_datasets['train'].classes


# Normalization parameters.
# See https://pytorch.org/docs/stable/torchvision/models.html for these values
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)


def determine_image_size(model_name):
    """Get the input size for a particlar model. These values are hardcoded
    and based on the code frameworks. Possibly subject to change.

    Arguments:
        model_name {string} -- name of the model

    Returns:
        int -- input size in pixels
    """
    try:
        return MODEL_TO_IMAGE_SIZE[model_name.lower()]
    except KeyError:
        return DEFAULT_IMAGE_SIZE


def make_transform_dict(image_size=DEFAULT_IMAGE_SIZE):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomAffine(degrees=30, shear=30, scale=(1, 1.75)),
                transforms.CenterCrop(950),
                transforms.RandomResizedCrop(image_size),
                # data augmentation, randomly flip and vertically flip across epochs
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # ColorJitter values chosen somewhat arbitrarily by what "looked" good
                # possibly something to optimize
                # transforms.ColorJitter(brightness=0.20, saturation=0.70, contrast=0.5,
                #                        hue=0.10),
                # convert to PyTorch tensor
                transforms.ToTensor(),
                # normalize
                transforms.Normalize(means, stds),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ]
        ),
    }
    # set test and val to same transform
    data_transforms["test"] = data_transforms["val"]
    return data_transforms


def get_dataset_sizes(dataloaders):
    """Get the sizes of datasets from dataloaders

    Arguments:
        dataloaders {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    dataset_sizes = {  # sizes for our progress bar
        phase: len(loader.dataset)
        for phase, loader in dataloaders.items()
    }
    return dataset_sizes


def get_dataloaders(data_dir, subsets, include_paths=False,
                    image_size=DEFAULT_IMAGE_SIZE, batch_size=4, shuffle=True):

    _, dataloaders = get_datasets_and_loaders(**locals())

    # # set the dataloader length to be the length of the dataset
    # for subset, dataloader in dataloaders.items():
    #     dataloader.__len__ = len(datasets[subset]) / batch_size
    return dataloaders


def get_datasets_and_loaders(data_dir, subsets, include_paths=False,
                             image_size=DEFAULT_IMAGE_SIZE, batch_size=4, shuffle=True):
    """Get dataset and DataLoader for a given data root directory
    Arguments:
        data_dir {string} -- path of directory
        subsets {tuple} -- contains "train", "val", or "test"; pass in multiple if
                                desired.

    Keyword Arguments:
        include_paths {bool} -- Whether to include file paths in the returned
                                dataset (default: {False})

    Returns:
        tuple -- datasets, dataloaders
    """
    if type(subsets) is str:
        subsets = (subsets, )

    data_transforms = make_transform_dict(image_size)

    # the dataset we use is either the normal ImageFolder, or our custom
    #   ImageFolder
    im_folder_class = ImageFolderWithPaths if include_paths else datasets.ImageFolder
    # get the datasets for each given subset, e.g. train, val, test
    image_datasets = {
        subset: im_folder_class(os.path.join(
            data_dir, subset), data_transforms[subset])
        for subset in subsets
    }
    # make dataloaders for each of the datasets above
    num_gpus = torch.cuda.device_count()
    dataloaders = {
        subset: torch.utils.data.DataLoader(
            image_datasets[subset],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_gpus * 4,
        )
        for subset in subsets
    }

    return image_datasets, dataloaders


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom Dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def make_csv_with_header(results_filepath, header):
    """Creates a csv file (overwrites if existing) for recording prediction
    results using the load_dir path specified in argparse.
    Writes the specified header to the top of the output file.

    Arguments:
        results_filepath {string} -- path of file to output csv
        header {string} -- the header to write

    Returns:
        string -- path of the created file
    """
    # write the csv header
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    with open(results_filepath, "w") as f:
        f.write(header + "\n")
    return results_filepath


def fit_model_last_to_dataset(model, dataset):
    num_out = len(dataset.classes)
    try:
        # case of torchvision
        num_in = model.fc.in_features
        model.fc = torch.nn.Linear(num_in, num_out)
    except AttributeError:
        try:
            # case of pretrainedmodels
            num_in = model.last_linear.in_features
            model.last_linear = torch.nn.Linear(num_in, num_out)
        except AttributeError as e:
            raise e

    model_name = get_model_name_from_fn(model)
    if "inception" in model_name:
        model.aux_logits = False

    return model


def get_model_name_from_fn(model_fn):
    return str(model_fn).split()[1]


def make_dataloaders_from_model_input_size(datadir, model_fn, batch_size):
    model_name = get_model_name_from_fn(model_fn)
    image_size = determine_image_size(model_name)
    return get_dataloaders(datadir, ('train', 'val'), image_size=image_size,
                           batch_size=batch_size)


def build_model_from_args(model_args):
    # split into model name and model_kwargs
    model_name, model_kwargs = args_to_name_and_kwargs(model_args)
    # Load the model
    # first try from torchvision
    try:
        model_fn = getattr(pretrainedmodels, model_name)
        print(f"Model {model_name} found under library pretrainedmodels.")
    except AttributeError:
        print(
            f"Could not find model {model_name} under pretrainedmodels. "
            + "Looking under torchvision.models."
        )
        try:
            model_fn = getattr(torchvision.models, model_name)
            print(f"Model {model_name} found under torchvision.models.")
        # else try from custom models
        except AttributeError:
            print(
                f"Could not find model {model_name} under torchvision.models. "
                + "Looking under src.custom_models."
            )
            try:
                model_fn = getattr(src.custom_models, model_name)
                print(f"Model {model_name} found under src.custom_models.")
            # else error
            except AttributeError as e:
                raise e
    # instatiate with kwargs
    model = model_fn(**model_kwargs)
    return model


def build_attr(module, attr_args=None, first_arg=None):
    if attr_args is None:
        return None
    attr_name, attr_kwargs = args_to_name_and_kwargs(attr_args)
    attr_fn = getattr(module, attr_name)
    attribute = (
        attr_fn(first_arg, **attr_kwargs) if first_arg else attr_fn(**attr_kwargs)
    )
    return attribute


def args_to_name_and_kwargs(model_and_kwargs_list):
    name = model_and_kwargs_list[0]
    kwargs = model_and_kwargs_list[1:]
    kwargs = make_kwargs_dict(kwargs)
    return name, kwargs


def make_kwargs_dict(kwargs_list):
    """Transforms a list of keyword arguments into a keyword arguments
    dictionary. Splits on the "=" symbol.
    E.g. ["first=1", "second=2", "third=3"] gets transformed into
    {"first": 1, "second": 2, "third": 3}.

    Arguments:
        kwargs_list {list} -- list of keyword arguments

    Returns:
        dict -- dictionary of keyword arguments
    """
    kwargs_dict = {
        # key -> value converted as the correct type
        kwarg.split("=")[0]: convert(kwarg.split("=")[1])
        for kwarg in kwargs_list
    }
    return kwargs_dict
