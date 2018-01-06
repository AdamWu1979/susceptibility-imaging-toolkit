import numpy as np
from scipy.ndimage import measurements as m, generate_binary_structure, binary_fill_holes

def imscale(image):
    max_v = np.max(image)
    min_v = np.min(image)
    return (image - min_v) / (max_v - min_v)

def create_mask(magnitude, level = 0.1):
    """Creates mask for a brain image assuming that the largest region is background.
    Steven Cao
    Hongjiang Wei
    University of California, Berkeley
    """
    magnitude = imscale(magnitude)
    if len(magnitude.shape) == 3:
        magnitude = magnitude.reshape(tuple(list(magnitude.shape)+[1]))
    mask_holder = []
    for echo in range(magnitude.shape[3]):
        image_mask = magnitude[:,:,:,echo] > level
        image_labels, num_labels = m.label(image_mask, generate_binary_structure(3,3))
        #stats = regionprops(image_labels, cache = False)
        #region_areas = [r.area for r in stats]
        region_areas = [np.sum(image_labels == i) for i in range(num_labels + 1)]
        largest_region = np.where(region_areas == np.max(region_areas))[0][0]
        image_mask = np.array(image_labels != largest_region)
        image_mask = binary_fill_holes(image_mask)
        #Fill mask
        
        #stats = regionprops(image_mask, cache = False)[0]
        #filled_image = stats.filled_image
        #box = stats.bbox
        mask_holder.append(image_mask)
    return np.stack(mask_holder, axis=3)
