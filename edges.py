import cv2
import numpy as np
import pywt

img = cv2.imread("sat_map3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# thick edge detection mask
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_anti_diagonal = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
kernel_main_diagonal = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
kernelx_reversed = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
kernely_reversed = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_anti_diagonal_reversed = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
kernel_main_diagonal_reversed = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])

# thin edge detection mask
horizontal = np.array([[-1, -1, -1], [1, 1, 1], [-1, -1, -1]])
vertical = np.array([[-1, 1, -1], [-1, 1, -1], [-1, 1, -1]])
anti_diagonal = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
main_diagonal = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
horizontal_negated = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 1]])
vertical_negated = np.array([[1, -1, 1], [1, -1, 1], [1, -1, 1]])
anti_diagonal_negated = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
main_diagonal_negated = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1]])

# applying thick edge detection mask
img_prewittx_reversed = cv2.filter2D(gray, -1, kernelx_reversed)
img_prewitty_reversed = cv2.filter2D(gray, -1, kernely_reversed)
img_prewitt_anti_diagonal_reversed = cv2.filter2D(
    gray, -1, kernel_anti_diagonal_reversed
)
img_prewitt_main_diagonal_reversed = cv2.filter2D(
    gray, -1, kernel_main_diagonal_reversed
)
img_prewittx = cv2.filter2D(gray, -1, kernelx)
img_prewitty = cv2.filter2D(gray, -1, kernely)
img_prewitt_anti_diagonal = cv2.filter2D(gray, -1, kernel_anti_diagonal)
img_prewitt_main_diagonal = cv2.filter2D(gray, -1, kernel_main_diagonal)

# applying thin edge detection mask
img_horizontal = cv2.filter2D(gray, -1, horizontal)
img_vertical = cv2.filter2D(gray, -1, vertical)
img_anti_diagonal = cv2.filter2D(gray, -1, anti_diagonal)
img_main_diagonal = cv2.filter2D(gray, -1, main_diagonal)
img_horizontal_negated = cv2.filter2D(gray, -1, horizontal_negated)
img_vertical_negated = cv2.filter2D(gray, -1, vertical_negated)
img_anti_diagonal_negated = cv2.filter2D(gray, -1, anti_diagonal_negated)
img_main_diagonal_negated = cv2.filter2D(gray, -1, main_diagonal_negated)

cv2.imshow(
    "Prewitt",
    img_prewittx
    + img_prewitty
    + img_prewitt_anti_diagonal
    + img_prewitt_main_diagonal
    + img_prewittx_reversed
    + img_prewitty_reversed
    + img_prewitt_anti_diagonal_reversed
    + img_prewitt_main_diagonal_reversed
    + img_horizontal
    + img_vertical
    + img_anti_diagonal
    + img_main_diagonal
    + img_horizontal_negated
    + img_vertical_negated
    + img_anti_diagonal_negated
    + img_main_diagonal_negated,
)

cv2.imshow("main", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
