import numpy as np


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
        print(temp_vec, x)

    return result


def custom_decomp_single_value(value, threshold_M):
    # Calculate the number of binary vectors
    total_bin_vectors = 2 * threshold_M

    # Define a placeholder for the matrix that will contain the results
    result = np.zeros((total_bin_vectors), dtype=np.int8)

    # Decompose a single value to m levels
    for m in range(-threshold_M + 1, threshold_M + 1):
        if value >= m:
            result[m + threshold_M - 1] = 1
        else:
            result[m + threshold_M - 1] = -1

    return result


if __name__ == "__main__":

    xdat = np.array([[0, 0, 2, -2, 1, 1, 0, -1, -1], [0, 0, 2, -2, 1, 1, 0, -1, -1]])

    res = custom_decomp_matrix(xdat, 2)

    print(xdat)
    print(res)
