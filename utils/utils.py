"""
Various functions used in the interactors.
"""
import numpy as np


def gray_to_rgb(im):
    """Convert a grayscale image to an RGB while keeping the same gray look.

    Parameter
    ---------
    im : array-like
        the grayscale image to convert

    Return
    ------
    im_rgb : array-like
        the RGB image

    """
    im_rgb = np.stack((im,) * 3, axis=-1)
    return im_rgb


def normalize_image(im):
    """Normalize an image into the range 0-255 as type np.uint8
    source: https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

    Parameters
    ----------
    im : array-like
        this should be a 2D image from the ome.tiff (or similar) that have a wide range of values. Don't pass a regular
        RGB image or an image that is already in the range of 0 to 255
    Return
    ------
    normalized_im : array-like
        the normalized image
    """
    pixels = im.flatten()

    # scale pixels to range 0 to 1
    normalized_im = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))

    # scale the pixels by 255
    normalized_im = (normalized_im.reshape(im.shape) * 255).astype(np.uint8)

    return normalized_im


def fix_segmentation_mask(mask):
    """Fixes segmentation masks by making the unique encoding go from 1 to number of objects present. I.e. if you have
    two objects in the mask then one object will be defined by the 1 int and the second object by the 2 int.

    Parameter
    ---------
    mask : array-like
        the object sementation mask

    Return
    ------
    fixed_mask : array-like
        the object segmentation mask, does not modify the mask in place

    """
    # object masks are defined by all pixels in a unique object containing the same value / label
    labels = np.unique(mask)
    fixed_mask = mask.copy()

    # there is a rare chance the mask will have no background (i.e. value of 0), be explicit about removing the 0 label
    if 0 in labels:
        labels = np.delete(labels, np.where(labels == 0)[0])

    for i, label in enumerate(labels):
        fixed_mask[mask == label] = i + 1

    return fixed_mask.astype(np.uint32)
