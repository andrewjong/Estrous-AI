import argparse
import utils

import numpy as np
import matplotlib.pyplot as plt

from utils import means, stds


parser = argparse.ArgumentParser(
    description="View the transforms established in utils.py on the train set")
parser.add_argument("data_dir",
                    help='Root directory of dataset to use, with classes \
                    separated into separate subdirectories.')
parser.add_argument("-n", "--number", default=20,
                    help="How many images to view (default: 20).")
args = parser.parse_args()

datasets, dataloaders = utils.get_datasets_and_loaders(
    args.data_dir, "train", include_paths=True, batch_size=1)

dataset, dataloader = datasets["train"], dataloaders["train"]


class DataScroller(object):
    def __init__(self, ax, images, labels, paths):
        self.ax = ax

        self.images, self.labels, self.paths = images, labels, paths
        self.num_samples, _, _, _ = images.shape
        self.ind = 0  # initialize start to zero

        self.im = ax.imshow(self.images[self.ind])
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if (hasattr(event, 'button') and event.button == 'up') or \
                (hasattr(event, 'key') and event.key == 'right'):
            self.ind = (self.ind + 1) % self.num_samples
        elif (hasattr(event, 'button') and event.button == 'down') or \
                (hasattr(event, 'key') and event.key == 'left'):
            self.ind = (self.ind - 1) % self.num_samples
        else:
            return
        self.update()

    def update(self):
        self.im.set_data(self.images[self.ind])
        ax.set_xlabel(f'Sample {self.ind}\n' + self.paths[self.ind])

        label_num = self.labels[self.ind]
        label_name = dataset.classes[label_num]
        ax.set_title(label_name, fontsize=16)
        self.im.axes.figure.canvas.draw()


fig, ax = plt.subplots(1, 1)
fig.suptitle("Use arrow keys or mouse wheel to scroll.")

# append to our nparrays
first_image_tensor, first_label_tensor, first_path = next(iter(dataloader))
images = first_image_tensor.numpy()  # convert tensor to numpy
labels = first_label_tensor.numpy()  # convert tensor to numpy
paths = np.asarray(first_path)       # convert tuple to numpy
for _ in range(args.number - 1):
    image_tensor, label_tensor, path_tuple = next(iter(dataloader))
    images = np.append(images, image_tensor.numpy(), 0)
    labels = np.append(labels, label_tensor.numpy())
    paths = np.append(paths, path_tuple[0])

# transpose tensor indexing to matplotlib indexing
# index, channels, w, h => index, w, h, channels
images = images.transpose((0, 2, 3, 1))

# unnormalize the images for matplotlib
images = np.array(stds) * images + np.array(means)
images = np.clip(images, 0, 1)

tracker = DataScroller(ax, images, labels, paths)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('key_press_event', tracker.onscroll)

plt.show()
