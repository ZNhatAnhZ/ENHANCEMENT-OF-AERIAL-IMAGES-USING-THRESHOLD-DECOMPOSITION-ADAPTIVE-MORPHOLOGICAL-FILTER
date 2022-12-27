import numpy as np


def decomposing_image(gray):
    normalized_gray_image = shifting_to_128(gray)
    array_of_binary_images = custom_decomp_matrix(normalized_gray_image, 127)
    return array_of_binary_images


def summation_of_binary_images(array_of_binary_images):
    gray = summation(array_of_binary_images)
    result = revert_128_to_255(gray)
    return result


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


def custom_decomp_single_value(value, threshold_M):
    temp = np.arange(-threshold_M + 1, threshold_M + 1)
    result = np.where(temp <= value, 1, -1)
    return result


def summation(matrix):
    # Find the number of rows and columns of the first matrix.
    rows, cols = matrix[0].shape

    # Define a placeholder for the matrix that will contain the results
    result = np.zeros((rows, cols))

    # Create an iterator with access to row, col indices in the data matrix
    it = np.nditer(matrix, flags=["multi_index", "f_index"])

    # Iterate over each element in the data matrix and obtain its threshold decomposition
    for i in it:
        index_of_m, row, col = it.multi_index
        result[row, col] += i / 2

    return result.astype(int)


# if __name__ == "__main__":
#     xdat = np.array([[0, 0, 2, -2, 1, 1, 0, -1, -1], [0, 2, 1, -2, 1, 1, 0, -1, -1]])

#     res1 = custom_decomp_matrix(
#         xdat, 2
#     )  # threshold m = the highest absolute value of x[i]
#     res2 = summation(res1)

#     print(xdat)
#     print(res1)
#     print(res2)
