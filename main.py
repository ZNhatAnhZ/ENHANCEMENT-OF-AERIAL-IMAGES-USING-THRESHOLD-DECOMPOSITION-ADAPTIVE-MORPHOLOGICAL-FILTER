import cv2
import threshold_decomposition as td
import edge_detection as ed
import morphological_filter as mf
import numpy as np
import scipy as sc

image = cv2.imread("./sample_images/sample1.png")

red = image[:, :, 2]
green = image[:, :, 1]
blue = image[:, :, 0]

binary_images_red = td.decomposing_image(red)
print("done red")
# binary_images_green = td.decomposing_image(green)
# print("done green")
# binary_images_blue = td.decomposing_image(blue)
# print("done blue")

red_thick_edges = ed.find_thick_edges(binary_images_red)
print("done finding thick edges")

morphological_filtered_images = mf.morphological_filter_for_binary_images(
    red_thick_edges
)
print("done morphological filtered images")
print(np.sum(morphological_filtered_images, axis=0))

cv2.imshow("main", np.sum(morphological_filtered_images, axis=0))

cv2.waitKey(0)
cv2.destroyAllWindows()
