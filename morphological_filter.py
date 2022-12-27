import numpy as np
import scipy as sc


def morphological_filter_for_binary_images(array_of_binary_images):
    # create a placeholder result array
    result = np.zeros(array_of_binary_images.shape, dtype=np.int8)

    # apply morphological filter for every binary images
    for i in range(len(array_of_binary_images)):
        result[i] = morphological_filter(array_of_binary_images[i])

    return result


def morphological_filter(array):
    # apply dilation filter on the binary image
    dilated = np.where(
        sc.ndimage.binary_dilation(array).astype(array.dtype) == 0, -1, 1
    )
    # apply erosion filter on the binary image
    eroded = np.where(sc.ndimage.binary_erosion(array).astype(array.dtype) == 0, -1, 1)

    # get the average value of dilation and erosion
    avg = (dilated + eroded) / 2

    # if each value of array >= each value of avg, it will be dilated value, if not then it will be eroded value
    mask = array >= avg
    result = np.copy(dilated)
    result[mask] = eroded[mask]

    return result
