import cv2
import threshold_decomposition as td
import edge_detection as ed
import morphological_filter as mf
import numpy as np

image = cv2.imread("./sample_images/sample1.png")

print("extracting three color channels...")
red = image[:, :, 2]
green = image[:, :, 1]
blue = image[:, :, 0]
print("done extracting three color channels")

print("decomposing 3 gray images into binary images...")
binary_images_red = td.decomposing_image(red)
print("done decomposing the gray image from red channel")
binary_images_green = td.decomposing_image(green)
print("done decomposing the gray image from green channel")
binary_images_blue = td.decomposing_image(blue)
print("done decomposing the gray image from blue channel")

print("finding thick edges of red channel...")
red_thick_edges = ed.find_thick_edges(binary_images_red)
print("done finding thick edges of red channel")
print("finding thick edges of green channel...")
green_thick_edges = ed.find_thick_edges(binary_images_green)
print("done finding thick edges of green channel")
print("finding thick edges of blue channel...")
blue_thick_edges = ed.find_thick_edges(binary_images_blue)
print("done finding thick edges of blue channel")

print("finding thin edges of red channel...")
red_thin_edges = ed.find_thin_edges(binary_images_red)
print("done finding thin edges of red channel")
print("finding thin edges of green channel...")
green_thin_edges = ed.find_thin_edges(binary_images_green)
print("done finding thin edges of green channel")
print("finding thin edges of blue channel...")
blue_thin_edges = ed.find_thin_edges(binary_images_blue)
print("done finding thin edges of blue channel")

red_edges = red_thick_edges + red_thin_edges
green_edges = green_thick_edges + green_thin_edges
blue_edges = blue_thick_edges + blue_thin_edges

print("applying morphological filter for red channel...")
red_morphological_filtered_images = mf.morphological_filter_for_binary_images(red_edges)
print("done applying morphological filter for red channel")
print("applying morphological filter for green channel...")
green_morphological_filtered_images = mf.morphological_filter_for_binary_images(
    green_edges
)
print("done applying morphological filter for green channel")
print("applying morphological filter for blue channel...")
blue_morphological_filtered_images = mf.morphological_filter_for_binary_images(
    blue_edges
)
print("done applying morphological filter for blue channel")

print("performing summation of all binary images and stack all color channels...")
red_result = td.summation_of_binary_images(red_morphological_filtered_images)
green_result = td.summation_of_binary_images(green_morphological_filtered_images)
blue_result = td.summation_of_binary_images(blue_morphological_filtered_images)
result = np.dstack((blue_result, green_result, red_result))

cv2.imshow("main", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
