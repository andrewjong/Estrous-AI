import numpy as np
import cv2
from PIL import Image


class AdaptiveThreshold(object):
    """Applies adaptive thresholding from OpenCV library. """

    def __init__(
        self, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, block_size=25, constant=2
    ):
        """Initializer. See cv2.adaptiveThreshold method for args
        
        Keyword Arguments:
            adaptiveMethod {[type]} -- [description]
                (default: {cv2.ADAPTIVE_THRESH_MEAN_C})
            block_size {int} -- [description] (default: {25})
            constant {int} -- [description] (default: {2})
        """
        self.adaptiveMethod = adaptiveMethod
        self.block_size = block_size
        self.constant = constant

    def __call__(self, img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = cv2.adaptiveThreshold(
            np_img,
            255,
            self.adaptiveMethod,
            cv2.THRESH_BINARY,
            self.block_size,
            self.constant,
        )
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img
