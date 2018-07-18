import numpy as np
from scipy.ndimage import measurements as m
from scipy.ndimage import generate_binary_structure, binary_fill_holes
from skimage.measure import regionprops

from functions import imscale, smv_filter

def create_mask(mag, level):
    """ Creates mask using thresholding.
    level: typically around 0.03 - 0.1
    """
    mag = imscale(np.abs(mag))
    mask = mag > level
    mask = polish_mask(mask)
    return mask

def polish_mask(mask):
    """Chooses the largest connected structure and fills holes."""
    labels, num_labels = m.label(mask, generate_binary_structure(3,3))
    props = regionprops(labels, cache = False)
    biggest = np.argmax([r.area for r in props]) + 1
    mask = labels == biggest
    mask = binary_fill_holes(mask)
    return mask

def create_mask_qsm(mag, voxel_size, level = 0.04):
    """Creates mask for qsm using thresholding and spherical-mean-value
    filtering. Use on echo closest to 30 ms.
    """
    mask = create_mask(mag, level)
    mask = smv_filter(mask, 2, voxel_size)
    mask = polish_mask(mask > 0.99)
    mask = smv_filter(mask, 3, voxel_size)
    mask = mask > 0.02
    mask = smv_filter(mask, 1, voxel_size)
    mask = mask > 0.97
    return mask