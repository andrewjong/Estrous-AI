import os

import torch
import PIL

from torchvision import datasets, transforms

# class_names = image_datasets['train'].classes

# Normalization parameters
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=30, shear=30, scale=(1, 1.75)),
        transforms.CenterCrop(950),
        transforms.RandomResizedCrop(224),
        # data augmentation, randomly flip and vertically flip across epochs
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # convert to PyTorch tensor
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(means, stds)
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
}
# set test and val to same transform
data_transforms['test'] = data_transforms['val']


def get_datasets_and_loaders(data_dir, *subsets, include_paths=False,
                             batch_size=4, shuffle=True):
    """Get dataset and DataLoader for a given data root directory
    Arguments:
        data_dir {string} -- path of directory
        subsets {string(s)} -- "train", "val", or "test"; pass in multiple if
                                desired.

    Keyword Arguments:
        include_paths {bool} -- Whether to include file paths in the returned
                                dataset (default: {False})

    Returns:
        tuple -- datasets, dataloaders
    """

    # the dataset we use is either the normal ImageFolder, or our custom
    #   ImageFolder
    im_folder_class = ImageFolderWithPaths if include_paths \
        else datasets.ImageFolder
    # get the datasets for each given subset, e.g. train, val, test
    image_datasets = {subset: im_folder_class(
        os.path.join(data_dir, subset), data_transforms[subset])
        for subset in subsets}
    # make dataloaders for each of the datasets above
    num_gpus = torch.cuda.device_count()
    dataloaders = {subset: torch.utils.data.DataLoader(
        image_datasets[subset], batch_size=batch_size, shuffle=shuffle,
        num_workers=num_gpus * 4)
        for subset in subsets}

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
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
