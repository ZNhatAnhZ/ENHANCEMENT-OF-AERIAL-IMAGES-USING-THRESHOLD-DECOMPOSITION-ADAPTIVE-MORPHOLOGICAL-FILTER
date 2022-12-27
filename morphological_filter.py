import numpy as np


def morphological_filter_for_binary_images(array_of_binary_images):
    result = np.zeros(array_of_binary_images.shape, dtype=np.uint8)

    for i in range(len(array_of_binary_images)):
        result[i] = morphological_filter(array_of_binary_images[i])

    return result


def morphological_filter(array):
    # padding the beginning and the ending of the x-axis of the array
    padded_array_dilated = np.pad(
        array, [(0, 0), (1, 1)], mode="constant", constant_values=0
    )
    padded_array_eroded = np.pad(
        array, [(0, 0), (1, 1)], mode="constant", constant_values=255
    )

    # get rows and columns of the padded array
    rows, cols = padded_array_dilated.shape

    # create result array to hold final value
    result = np.zeros((rows, cols - 2), dtype=np.uint8)

    # iterate through the array and perform dilation
    for i in range(0, rows):
        for j in range(1, cols - 1):
            dilated_value = int(
                max(
                    padded_array_dilated[i][j - 1],
                    padded_array_dilated[i][j],
                    padded_array_dilated[i][j + 1],
                )
            )
            eroded_value = int(
                min(
                    padded_array_eroded[i][j - 1],
                    padded_array_eroded[i][j],
                    padded_array_eroded[i][j + 1],
                )
            )
            avg_of_dilated_eroded = (dilated_value + eroded_value) / 2
            if padded_array_dilated[i][j] >= avg_of_dilated_eroded:
                result[i][j - 1] = dilated_value
            else:
                result[i][j - 1] = eroded_value

    return result
