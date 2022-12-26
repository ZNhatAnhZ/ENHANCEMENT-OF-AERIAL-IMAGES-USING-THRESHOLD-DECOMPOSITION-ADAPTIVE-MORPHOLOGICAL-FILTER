import numpy as np


def morphological_filter(array):
    # padding the beginning and the ending of the x-axis of the array
    padded_array = np.pad(array, [(0, 0), (1, 1)], mode="constant", constant_values=0)

    # get rows and columns of the padded array
    rows, cols = padded_array.shape

    # create result array to hold final value
    result = np.zeros((rows, cols - 2), dtype=np.uint8)

    # iterate through the array and perform dilation
    for i in range(0, rows):
        for j in range(1, cols - 1):
            dilated_value = int(
                max(padded_array[i][j - 1], padded_array[i][j], padded_array[i][j + 1])
            )
            eroded_value = int(
                min(
                    padded_array[i][j - 1],
                    padded_array[i][j],
                    padded_array[i][j + 1],
                )
            )
            avg_of_dilated_eroded = (dilated_value + eroded_value) / 2
            if padded_array[i][j] >= avg_of_dilated_eroded:
                result[i][j - 1] = dilated_value
            else:
                result[i][j - 1] = eroded_value

    return result
