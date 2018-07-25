import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

import cv2
from torchvision import datasets, transforms

# class_names = image_datasets['train'].classes

# Normalization parameters
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)


class AdaptiveThreshold(object):
    """Applies adaptive thresholding from OpenCV library. """

    def __init__(self, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                 block_size=25, constant=2):
        """Initializer. See cv2.adaptiveThreshold method for args
        
        Keyword Arguments:
            adaptiveMethod {[type]} -- [description] (default: {cv2.ADAPTIVE_THRESH_MEAN_C})
            block_size {int} -- [description] (default: {25})
            constant {int} -- [description] (default: {2})
        """
        self.adaptiveMethod = adaptiveMethod
        self.block_size = block_size
        self.constant = constant

    def __call__(self, img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = cv2.adaptiveThreshold(np_img, 255, self.adaptiveMethod,
                                       cv2.THRESH_BINARY, self.block_size,
                                       self.constant)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img


data_transforms = {
    'train': transforms.Compose([
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


def get_datasets_and_loaders(data_dir, *subsets, include_paths=False):
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
        image_datasets[subset], batch_size=4, shuffle=True,
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


if __name__ == '__main__':
    threshold_tsfm_1 = AdaptiveThreshold(cv2.ADAPTIVE_THRESH_MEAN_C, 25)
    threshold_tsfm_2 = AdaptiveThreshold(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    dataset = datasets.ImageFolder('data/remove_art/train')
    sample, _ = dataset[random.randrange(0, len(dataset))]
    transformed_sample_1 = threshold_tsfm_1(sample)
    transformed_sample_2 = threshold_tsfm_2(sample)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(sample)
    b = fig.add_subplot(1, 2, 2)
    plt.imshow(transformed_sample_1)
    plt.show()

# def show_first_inputs():
#     inputs, classes = next(iter(dataloaders["train"]))
#     inputs = torchvision.utils.make_grid(inputs)
#     imshow(inputs)


# def imshow(inputs, title=None):
#     """ Show a PyTorch Tensor as an image, after unnormalizing """
#     inputs = inputs.numpy().transpose((1, 2, 0))
#     mean = np.array(MEANS)
#     std = np.array(STDS)
#     # unnormalize
#     inputs = std * inputs + mean
#     # constrain values to be between 0 and 1, in case any went over
#     inputs = np.clip(inputs, 0, 1)
#     plt.imshow(inputs)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # let plots update


# def visualize_model(model, num_images=6):
#     """Visualize the model's current predictions on some images.

#     Arguments:
#         model {[type]} -- [description]

#     Keyword Arguments:
#         num_images {int} -- [description] (default: {6})
#     """

#     was_training = model.training
#     # why do we do this?
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, predictions = torch.max(outputs, 1)

#             for j in range(inputs.size(0)):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[predictions[j]]}')
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return

#             model.train(mode=was_training)
