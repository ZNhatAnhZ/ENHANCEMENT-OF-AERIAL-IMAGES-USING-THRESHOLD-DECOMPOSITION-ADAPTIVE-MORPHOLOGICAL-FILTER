import numpy as np

# extract a gray image into an array of 127 binary images
def decomposing_image(gray):
    normalized_gray_image = shifting_to_128(gray)
    array_of_binary_images = custom_decomp_matrix(normalized_gray_image, 127)
    return array_of_binary_images


# we need to shift range from (0, 255) to (-128, 127) because we want to turn it into 2m binary signals
def shifting_to_128(gray):
    # Define a placeholder for the matrix that will contain the results
    result = np.zeros(gray.shape, dtype=np.int8)

    it = np.nditer(gray, flags=["multi_index"])
    for x in it:
        row, col = it.multi_index
        # if x == 0:
        #     x = x + 1
        result[row, col] = x - 128
    return result


# function for decomposing a gray image into binary images
def custom_decomp_matrix(matrix, threshold_M):
    # Find the number of rows and columns in the data matrix.
    rows, cols = matrix.shape

    # Calculate the number of binary vectors
    total_bin_vectors = 2 * threshold_M

    # Define a placeholder for the matrix that will contain the results
    result = np.zeros((total_bin_vectors, rows, cols), dtype=np.int8)

    # Create an iterator with access to row, col indices in the data matrix
    it = np.nditer(matrix, flags=["multi_index"])

    # Iterate over each element in the data matrix and obtain its threshold decomposition
    for x in it:
        row, col = it.multi_index
        temp_vec = custom_decomp_single_value(x, threshold_M)
        result[:, row, col] = temp_vec[::-1]

    return result


# function for decomposing a single value into different binary values
def custom_decomp_single_value(value, threshold_M):
    temp = np.arange(-threshold_M + 1, threshold_M + 1)
    result = np.where(temp <= value, 1, -1)
    return result


# sum of all binary images to reconstruct a gray image
def summation_of_binary_images(array_of_binary_images):
    gray = np.sum(array_of_binary_images, axis=0, dtype=np.int8) / 2
    return revert_128_to_255(gray)


# revert image from range (-128, 127) to (0, 255)
def revert_128_to_255(matrix):
    # Define a placeholder for the matrix that will contain the results
    result = np.zeros(matrix.shape, dtype=np.uint8)

    it = np.nditer(matrix, flags=["multi_index"])
    for x in it:
        row, col = it.multi_index
        # if x == 0:
        #     x = x + 1
        result[row, col] = x + 128
    return result
