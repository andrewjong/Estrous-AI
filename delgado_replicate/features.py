from math import radians as rad
from typing import List, Tuple

import numpy as np
import skimage

from skimage.feature import greycomatrix, greycoprops



def otsu_segment(image, median_filter_blur=7, inplace=False):
    """
    Runs a median filter and otsu thresholding to segment images. Image must be single channel uint8.
    Returns the segmented image, as well as the mask.
    For the mask, 1 is areas of interest, 0 is background.

    Args:
        image:
        median_filter_blur:
        inplace:

    Returns:

    """
    assert len(image.shape) == 2, "Must be a 2-D image"
    # whether to do inplace edit or not
    if not inplace:
        image = image.copy()
    # turn to grayscale
    # apply median blur
    blur = np.ones((median_filter_blur, median_filter_blur))
    denoised = skimage.filters.median(image, blur)
    # segment using otsu threshold
    thresh = skimage.filters.threshold_otsu(denoised)
    mask = denoised > thresh
    image[mask] = 0
    # invert the mask before we used it to "select" the background. but we want to
    # return 1 at the area of interest
    return image, np.invert(mask).astype(int)


def extract_shape_factors(otsu_mask):
    """
    Extracts "morphological metrics" from Delgado et al. 2017, bottom left column on
    page 3.

    Args:
        otsu_mask: a 2-D numpy array to represent an image.

    Returns:
        A 3-tuple, with (1) number of connected elements (2) average compactness (3) total compactness

    """
    connected_labels = skimage.measure.label(otsu_mask)
    # add +1 to connected labels, because 0 value is ignored
    rprops = skimage.measure.regionprops(connected_labels + 1)
    num_elements = len(rprops)

    # def a function
    compactness_factor = (
        lambda rprop: rprop.area / rprop.perimeter ** 2 if rprop.perimeter > 0 else 0
    )

    total_compactness = 0
    for r in rprops:
        fc_i = compactness_factor(r)
        total_compactness += fc_i

    average_compactness = total_compactness / num_elements

    return num_elements, average_compactness, total_compactness


def extract_cielab_channels(image):
    """
    Extracts channels used by Delgado et al. 2017: grayscale and alpha and beta from CIELAB

    Args:
        image: RGB image

    Returns:
        gray, alpha, beta (all 2D arrays)
    """

    grayed = skimage.color.rgb2gray(image)
    grayed = skimage.img_as_ubyte(grayed)  # convert [0,1] to [0,255]
    labbed = skimage.color.rgb2lab(image)
    # to convert to gray, alpha + 127, beta + 128. https://stackoverflow.com/questions/25294141/cielab-color-range-for-scikit-image
    alpha = (labbed[:, :, 1] + 127).astype("uint8")
    beta = (labbed[:, :, 2] + 128).astype("uint8")

    return grayed, alpha, beta


def extract_delgado_features(
    image,
    props=("contrast", "dissimilarity", "homogeneity", "energy"),
    distances=(1, 5),
    angles=(0, rad(45), rad(90), rad(135)),
) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    """
    Extracts GLCM and morphological features according to Delgado et al. 2017 from a
    given image

    Args:
        image: a 3-channel RGB image
        props: list of desired properties to extract via skimage.feature.greycoprops
        distances:
        angles:

    Returns:
        (1) 1-d numpy array of extracted features. (2) 3-tuple, the segmented images from gray, alpha, beta extracted channels.

    """

    grayed, alpha, beta = extract_cielab_channels(image)

    segmentations = []
    feature_arrs: List[np.ndarray] = []
    for channel in grayed, alpha, beta:
        segmented, mask = otsu_segment(channel)
        segmentations.append(segmented)
        glcm = greycomatrix(segmented, distances, angles, normed=True)

        for prop in props:
            feat = greycoprops(glcm, prop)
            # feat is a 2d array: distance, angle. need to flatten
            feature_arrs.append(np.ravel(feat))

        morphological = extract_shape_factors(mask)
        feature_arrs.append(morphological)

    features = np.concatenate(feature_arrs)
    return features, segmentations
