import os

import torch
from torchvision import datasets, models, transforms

# class_names = image_datasets['train'].classes

# Normalization parameters
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

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


def get_datasets_and_loaders(data_dir, *subsets):
    """Get dataset and DataLoader for a given data root directory.

    Arguments:
        data_dir {string} -- path of directory
        subsets {string(s)} -- "train", "val", or "test"; pass in multiple if desired.

    Returns:
        [type] -- [description]
    """

    image_datasets = {subset: datasets.ImageFolder(os.path.join(data_dir, subset),
                                                   data_transforms[subset])
                      for subset in subsets}

    dataloaders = {subset: torch.utils.data.DataLoader(
        image_datasets[subset], batch_size=4, shuffle=True, num_workers=0)
        for subset in subsets}

    return image_datasets, dataloaders


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