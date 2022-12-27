import cv2
import numpy as np
import scipy as sc

g = np.array([0, 0, 0], dtype=np.int8)  # structuring elememt

# thick edge detection mask
kernelx = np.array([[1, 1, 1], [g[0], g[1], g[2]], [-1, -1, -1]])
kernely = np.array([[-1, g[0], 1], [-1, g[1], 1], [-1, g[2], 1]])
kernel_anti_diagonal = np.array([[g[0], 1, 1], [-1, g[1], 1], [-1, -1, g[2]]])
kernel_main_diagonal = np.array([[-1, -1, g[0]], [-1, g[1], 1], [g[2], 1, 1]])
kernelx_reversed = np.array([[-1, -1, -1], [g[0], g[1], g[2]], [1, 1, 1]])
kernely_reversed = np.array([[1, g[0], -1], [1, g[1], -1], [1, g[2], -1]])
kernel_anti_diagonal_reversed = np.array([[g[0], -1, -1], [1, g[1], -1], [1, 1, g[2]]])
kernel_main_diagonal_reversed = np.array([[1, 1, g[0]], [1, g[1], -1], [g[2], -1, -1]])

# thin edge detection mask
horizontal = np.array([[-1, -1, -1], [1, 1, 1], [-1, -1, -1]])
vertical = np.array([[-1, 1, -1], [-1, 1, -1], [-1, 1, -1]])
anti_diagonal = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
main_diagonal = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
horizontal_negated = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 1]])
vertical_negated = np.array([[1, -1, 1], [1, -1, 1], [1, -1, 1]])
anti_diagonal_negated = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
main_diagonal_negated = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1]])


def find_thick_edges(array_of_binary_images):
    result = np.zeros(array_of_binary_images.shape, dtype=np.uint8)
    # applying thick edge detection mask
    for i in range(len(array_of_binary_images)):
        img_prewittx = sc.signal.convolve2d(
            array_of_binary_images[i], kernelx, mode="same"
        )

        img_prewitty = sc.signal.convolve2d(
            array_of_binary_images[i], kernely, mode="same"
        )

        img_prewitt_anti_diagonal = sc.signal.convolve2d(
            array_of_binary_images[i], kernel_anti_diagonal, mode="same"
        )

        img_prewitt_main_diagonal = sc.signal.convolve2d(
            array_of_binary_images[i], kernel_main_diagonal, mode="same"
        )

        img_prewittx_reversed = sc.signal.convolve2d(
            array_of_binary_images[i], kernelx_reversed, mode="same"
        )

        img_prewitty_reversed = sc.signal.convolve2d(
            array_of_binary_images[i], kernely_reversed, mode="same"
        )

        img_prewitt_anti_diagonal_reversed = sc.signal.convolve2d(
            array_of_binary_images[i], kernel_anti_diagonal_reversed, mode="same"
        )
        img_prewitt_main_diagonal_reversed = sc.signal.convolve2d(
            array_of_binary_images[i], kernel_main_diagonal_reversed, mode="same"
        )
        result[i] = (
            img_prewittx
            + img_prewitty
            + img_prewitt_anti_diagonal
            + img_prewitt_main_diagonal
            + img_prewittx_reversed
            + img_prewitty_reversed
            + img_prewitt_anti_diagonal_reversed
            + img_prewitt_main_diagonal_reversed
        )

    return result

def find_thick_edges(array_of_binary_images):
    result = np.zeros(array_of_binary_images.shape, dtype=np.uint8)
    # # applying thick edge detection mask
    # for i in range(len(array_of_binary_images)):
    #     img_prewittx = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernelx, mode="same"
    #     )

    #     img_prewitty = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernely, mode="same"
    #     )

    #     img_prewitt_anti_diagonal = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernel_anti_diagonal, mode="same"
    #     )

    #     img_prewitt_main_diagonal = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernel_main_diagonal, mode="same"
    #     )

    #     img_prewittx_reversed = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernelx_reversed, mode="same"
    #     )

    #     img_prewitty_reversed = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernely_reversed, mode="same"
    #     )

    #     img_prewitt_anti_diagonal_reversed = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernel_anti_diagonal_reversed, mode="same"
    #     )
    #     img_prewitt_main_diagonal_reversed = sc.signal.convolve2d(
    #         array_of_binary_images[i], kernel_main_diagonal_reversed, mode="same"
    #     )
    #     result[i] = (
    #         img_prewittx
    #         + img_prewitty
    #         + img_prewitt_anti_diagonal
    #         + img_prewitt_main_diagonal
    #         + img_prewittx_reversed
    #         + img_prewitty_reversed
    #         + img_prewitt_anti_diagonal_reversed
    #         + img_prewitt_main_diagonal_reversed
    #     )

    return result


# # applying thin edge detection mask
# img_horizontal = cv2.filter2D(gray, -1, horizontal)
# img_vertical = cv2.filter2D(gray, -1, vertical)
# img_anti_diagonal = cv2.filter2D(gray, -1, anti_diagonal)
# img_main_diagonal = cv2.filter2D(gray, -1, main_diagonal)
# img_horizontal_negated = cv2.filter2D(gray, -1, horizontal_negated)
# img_vertical_negated = cv2.filter2D(gray, -1, vertical_negated)
# img_anti_diagonal_negated = cv2.filter2D(gray, -1, anti_diagonal_negated)
# img_main_diagonal_negated = cv2.filter2D(gray, -1, main_diagonal_negated)

# all_edge_detected_images = np.array(
#     [
#         img_prewittx,
#         img_prewitty,
#         img_prewitt_anti_diagonal,
#         img_prewitt_main_diagonal,
#         img_prewittx_reversed,
#         img_prewitty_reversed,
#         img_prewitt_anti_diagonal_reversed,
#         img_prewitt_main_diagonal_reversed,
#         img_horizontal,
#         img_vertical,
#         img_anti_diagonal,
#         img_main_diagonal,
#         img_horizontal_negated,
#         img_vertical_negated,
#         img_anti_diagonal_negated,
#         img_main_diagonal_negated,
#     ]
# )

# edge_detected_image = (
#     img_prewittx
#     + img_prewitty
#     + img_prewitt_anti_diagonal
#     + img_prewitt_main_diagonal
#     + img_prewittx_reversed
#     + img_prewitty_reversed
#     + img_prewitt_anti_diagonal_reversed
#     + img_prewitt_main_diagonal_reversed
#     + img_horizontal
#     + img_vertical
#     + img_anti_diagonal
#     + img_main_diagonal
#     + img_horizontal_negated
#     + img_vertical_negated
#     + img_anti_diagonal_negated
#     + img_main_diagonal_negated
# )

# img = cv2.imread("./sample_images/sat_map3.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
